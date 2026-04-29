"""
OcclusionMask Node for ComfyUI
-------------------------------
Purpose: Generate an occlusion protection mask for use with ReActor (and similar
faceswap nodes). The mask marks pixels that are covered by objects IN FRONT OF
the face (microphones, hands, glasses, food, etc.) so the faceswap node knows
to leave those pixels untouched.

Pipeline per face:
  1. Detect face(s) with InsightFace
  2. Crop + pad the face region, resize to 256x256
  3. Run occluder.onnx  → "something is in front of the face here"
  4. Run XSeg_model.onnx → "this is the face skin region"
  5. Final mask = occluder_mask AND NOT xseg_face_mask
     = "pixels that are occluded AND not naked face skin"
  6. Post-process: expand → feather
  7. Paste mask back onto full-image canvas
  8. Repeat for every detected face, union all per-face masks

ReActor wiring:
  IMAGE  → ReActor  input_image
  MASK   → ReActor  face_mask  (tells ReActor: "preserve these pixels")
  PREVIEW → Preview Image node (for tuning / debugging)

Author: redesigned for clarity and correctness.
"""

import os
import logging
import numpy as np
from PIL import Image, ImageDraw
import torch
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from .face_helpers.face_masks import FaceMasks

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Providers helper
# ---------------------------------------------------------------------------
def _ort_providers():
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )


# ---------------------------------------------------------------------------
# Model singleton cache (module-level so they survive across node executions)
# ---------------------------------------------------------------------------
_occluder_sess: ort.InferenceSession | None = None
_xseg_sess:     ort.InferenceSession | None = None
_face_detector: FaceAnalysis | None = None


def _get_occluder_session() -> ort.InferenceSession:
    global _occluder_sess
    if _occluder_sess is None:
        path = os.path.join(os.path.dirname(__file__), "models", "occluder.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"occluder.onnx not found at: {path}")
        _occluder_sess = ort.InferenceSession(path, providers=_ort_providers())
        _logger.info("Loaded occluder.onnx")
    return _occluder_sess


def _get_xseg_session() -> ort.InferenceSession:
    global _xseg_sess
    if _xseg_sess is None:
        path = os.path.join(os.path.dirname(__file__), "models", "XSeg_model.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"XSeg_model.onnx not found at: {path}")
        _xseg_sess = ort.InferenceSession(path, providers=_ort_providers())
        _logger.info("Loaded XSeg_model.onnx")
    return _xseg_sess


def _get_face_detector() -> FaceAnalysis:
    global _face_detector
    if _face_detector is None:
        # Configure INSIGHTFACE_HOME — scan common locations
        if not os.environ.get("INSIGHTFACE_HOME"):
            candidates = [
                os.path.join(os.path.dirname(__file__), "models", "insightface"),
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "insightface"),
                os.path.join(os.path.expanduser("~"), ".insightface"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    os.environ["INSIGHTFACE_HOME"] = p
                    _logger.info(f"Set INSIGHTFACE_HOME → {p}")
                    break
            else:
                _logger.warning(
                    "INSIGHTFACE_HOME not found in common paths. "
                    "Set INSIGHTFACE_HOME environment variable if face detection fails."
                )

        _face_detector = FaceAnalysis(providers=_ort_providers())
        _face_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        _logger.info("Loaded InsightFace detector")
    return _face_detector


# ---------------------------------------------------------------------------
# Low-level mask inference helpers
# ---------------------------------------------------------------------------

def _run_occluder(sess: ort.InferenceSession, img_256_chw_float01: np.ndarray) -> np.ndarray:
    """
    img_256_chw_float01: float32 array [3, 256, 256] in range [0, 1]
    Returns: float32 [256, 256] binary mask where 1 = occluder present
    """
    inp = img_256_chw_float01[None]  # [1, 3, 256, 256]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    mask = out.squeeze().astype(np.float32)
    mask = (mask > 0).astype(np.float32)
    return mask  # [256, 256]


def _run_xseg(sess: ort.InferenceSession, img_256_chw_float01: np.ndarray) -> np.ndarray:
    """
    img_256_chw_float01: float32 array [3, 256, 256] in range [0, 1]
    Returns: float32 [256, 256] mask where 1 = face skin (NOT the object)
    XSeg raw output is "object/non-face" so we invert it to get the face region.
    """
    inp = img_256_chw_float01[None]  # [1, 3, 256, 256]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    mask = out.squeeze().astype(np.float32)
    mask = np.clip(mask, 0.0, 1.0)
    mask[mask < 0.1] = 0.0
    # Raw XSeg output: high value = non-face/object → invert for face skin
    face_mask = 1.0 - mask
    return face_mask  # [256, 256], 1 = face skin


# ---------------------------------------------------------------------------
# Per-face mask computation
# ---------------------------------------------------------------------------

def _compute_face_occluder_mask(
    face_crop_rgb: np.ndarray,       # uint8 [H, W, 3]
    detection_sensitivity: float,    # 0.0–1.0, higher = more objects detected
    mask_mode: str,                  # "Hard" | "Soft"
    mask_expansion: int,             # pixels to expand mask outward (can be negative)
    edge_softness: int,              # gaussian blur radius after expansion
    occluder_sess: ort.InferenceSession,
    xseg_sess: ort.InferenceSession,
) -> np.ndarray:
    """
    Returns a float32 [256, 256] mask in range [0,1].
    1 = occluded pixel (object in front of face) → faceswap should NOT touch this.
    0 = clear face pixel → faceswap can swap here.
    """
    # Resize to 256×256 and normalize to [0, 1]
    resized = cv2.resize(face_crop_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_f = resized.astype(np.float32) / 255.0           # [256, 256, 3]
    img_chw = img_f.transpose(2, 0, 1)                   # [3, 256, 256]

    # --- Model inference ---
    occluder_mask = _run_occluder(occluder_sess, img_chw)  # 1 = something covering face
    face_skin_mask = _run_xseg(xseg_sess, img_chw)        # 1 = face skin

    # --- Combine: we want pixels that are OCCLUDED and NOT face skin ---
    # Threshold for sensitivity: lower threshold = more objects detected
    # detection_sensitivity 1.0 → threshold 0.1 (very sensitive, catches more objects)
    # detection_sensitivity 0.0 → threshold 0.9 (strict, only very clear occlusions)
    threshold = 1.0 - (detection_sensitivity * 0.8 + 0.1)

    # Soft-threshold occluder mask
    combined = np.clip(occluder_mask - face_skin_mask * 0.5, 0.0, 1.0)

    if mask_mode == "Hard":
        combined = (combined >= threshold).astype(np.float32)
    else:
        # Soft mode: keep the gradient but suppress below threshold
        combined = np.where(combined < threshold * 0.5, 0.0, combined)
        combined = np.clip(combined, 0.0, 1.0)

    # --- Mask expansion (morphological dilation/erosion) ---
    if mask_expansion != 0:
        abs_exp = abs(mask_expansion)
        # Use ellipse kernel for natural-looking edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * abs_exp + 1, 2 * abs_exp + 1)
        )
        if mask_expansion > 0:
            combined = cv2.dilate(combined, kernel, iterations=1)
        else:
            combined = cv2.erode(combined, kernel, iterations=1)
        combined = np.clip(combined, 0.0, 1.0)

    # --- Edge softness (always applied AFTER expansion) ---
    if edge_softness > 0:
        ksize = edge_softness * 2 + 1
        combined = cv2.GaussianBlur(combined, (ksize, ksize), 0)

    return combined  # float32 [256, 256]


# ---------------------------------------------------------------------------
# Debug preview image builder
# ---------------------------------------------------------------------------

def _build_preview(
    orig_rgb: np.ndarray,    # uint8 [H, W, 3]
    full_mask: np.ndarray,   # float32 [H, W] in [0, 1]
    face_bboxes: list,       # list of (x1,y1,x2,y2) in original image coords
) -> np.ndarray:
    """
    Returns uint8 [H, W, 3] image:
    - Red semi-transparent overlay on masked (protected) region
    - Green bounding boxes around detected faces
    """
    preview = orig_rgb.copy().astype(np.float32)
    H, W = orig_rgb.shape[:2]

    # Red overlay for occluded/protected region
    red_layer = np.zeros_like(preview)
    red_layer[..., 0] = 255.0  # R channel

    alpha = full_mask[..., None] * 0.6  # 60% opacity on masked area
    preview = preview * (1.0 - alpha) + red_layer * alpha
    preview = np.clip(preview, 0, 255).astype(np.uint8)

    # Green face bounding boxes
    for (x1, y1, x2, y2) in face_bboxes:
        cv2.rectangle(preview, (x1, y1), (x2, y2), color=(0, 220, 60), thickness=2)

    return preview


# ---------------------------------------------------------------------------
# Tensor ↔ numpy helpers
# ---------------------------------------------------------------------------

def _tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Accepts ComfyUI IMAGE tensor of shape [1, H, W, 3] or [H, W, 3].
    Returns uint8 numpy array [H, W, 3].
    """
    arr = tensor.cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr  # uint8 [H, W, 3]


def _numpy_rgb_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H, W, 3] → float32 tensor [1, H, W, 3]"""
    f = arr.astype(np.float32) / 255.0
    return torch.from_numpy(f).unsqueeze(0)  # [1, H, W, 3]


def _mask_numpy_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """float32 [H, W] → float32 tensor [1, H, W]"""
    return torch.from_numpy(mask[None, ...].copy()).float()


# ---------------------------------------------------------------------------
# Main node class
# ---------------------------------------------------------------------------

class OcclusionMask:
    """
    ComfyUI node: generates an occlusion protection mask for ReActor faceswap.

    The MASK output marks pixels that are covered by objects in front of the face
    (microphones, hands, glasses, food, etc.).  Feed this mask into ReActor's
    'face_mask' input — ReActor will skip those pixels and preserve the occlusion.

    Supports batch IMAGE inputs (list of [1,H,W,3] tensors or a single [N,H,W,3]).
    """

    # Singleton model cache — loaded once, reused across all executions
    _occluder_sess: ort.InferenceSession | None = None
    _xseg_sess:     ort.InferenceSession | None = None
    _face_detector: FaceAnalysis | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Input ──────────────────────────────────────────────────
                "input_image": (
                    "IMAGE",
                    {"forceInput": True, "label": "Input Image (pass-through to ReActor)"},
                ),

                # ── Detection ──────────────────────────────────────────────
                "face_target": (
                    ["Largest face only", "All faces"],
                    {
                        "default": "Largest face only",
                        "label": "Face Target",
                        "tooltip": (
                            "Largest face only: fastest, best for single-subject images. "
                            "All faces: processes every detected face independently and unions the masks."
                        ),
                    },
                ),
                "face_crop_padding": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "display": "slider",
                        "label": "Face Crop Padding  (0–50%)",
                        "tooltip": (
                            "How much extra area around the detected face bounding box to include "
                            "before running the occlusion models. Increase if objects near the face "
                            "edges are being missed. 15% is a good default."
                        ),
                    },
                ),
                "fallback_to_full_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": "Fallback to Full Image if No Face Found",
                        "tooltip": (
                            "If enabled and no face is detected, the entire image is treated as "
                            "the face region. Useful for heavily occluded faces. "
                            "Disable for safety to avoid accidental full-image masking."
                        ),
                    },
                ),

                # ── Sensitivity ────────────────────────────────────────────
                "detection_sensitivity": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "label": "Detection Sensitivity  (0 = strict → 1 = detect everything)",
                        "tooltip": (
                            "Controls how aggressively objects are detected.\n"
                            "LOW (0.2–0.4): only very prominent occlusions like a hand covering the face.\n"
                            "MID (0.5–0.7): good default — catches microphones, glasses, held objects.\n"
                            "HIGH (0.8–1.0): catches subtle occlusions but may bleed into face edges."
                        ),
                    },
                ),

                # ── Mask shape ─────────────────────────────────────────────
                "mask_expansion": (
                    "INT",
                    {
                        "default": 6,
                        "min": -20,
                        "max": 50,
                        "step": 1,
                        "display": "slider",
                        "label": "Mask Expansion  (-20 shrink ↔ +50 grow, px)",
                        "tooltip": (
                            "Expands (positive) or shrinks (negative) the detected occlusion mask.\n"
                            "Positive values add a safety margin around objects — recommended to avoid "
                            "faceswap pixels bleeding under object edges.\n"
                            "Negative values trim the mask if it's accidentally covering face skin."
                        ),
                    },
                ),
                "edge_softness": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 30,
                        "step": 1,
                        "display": "slider",
                        "label": "Edge Softness  (0 = hard edge → 30 = very soft blend)",
                        "tooltip": (
                            "Gaussian blur applied to the mask AFTER expansion. "
                            "Soft edges create a smoother transition between the swapped face "
                            "and the preserved object, avoiding hard seams. "
                            "4–8px is recommended for most cases."
                        ),
                    },
                ),
                "mask_mode": (
                    ["Soft (recommended)", "Hard (binary)"],
                    {
                        "default": "Soft (recommended)",
                        "label": "Mask Mode",
                        "tooltip": (
                            "Soft: the mask retains gradient values (0.0–1.0) allowing partial "
                            "blending at object boundaries — best for natural-looking results.\n"
                            "Hard: mask is strictly 0 or 1 — use if ReActor produces ghosting "
                            "artifacts with soft masks."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES  = ("IMAGE", "MASK", "PREVIEW")
    FUNCTION      = "generate_occlusion_mask"
    CATEGORY      = "image/faceswap"

    # ------------------------------------------------------------------
    # Model accessors (lazy-load singletons)
    # ------------------------------------------------------------------

    @classmethod
    def _get_occluder(cls) -> ort.InferenceSession:
        if cls._occluder_sess is None:
            cls._occluder_sess = _get_occluder_session()
        return cls._occluder_sess

    @classmethod
    def _get_xseg(cls) -> ort.InferenceSession:
        if cls._xseg_sess is None:
            cls._xseg_sess = _get_xseg_session()
        return cls._xseg_sess

    @classmethod
    def _get_detector(cls) -> FaceAnalysis:
        if cls._face_detector is None:
            cls._face_detector = _get_face_detector()
        return cls._face_detector

    # ------------------------------------------------------------------
    # Face detection helpers
    # ------------------------------------------------------------------

    def _detect_faces(self, img_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Returns list of (x1, y1, x2, y2) bounding boxes, sorted largest-first."""
        try:
            faces = self._get_detector().get(img_rgb)
        except Exception as e:
            _logger.error(f"InsightFace detection failed: {e}")
            return []
        bboxes = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            # Clamp to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_rgb.shape[1], x2); y2 = min(img_rgb.shape[0], y2)
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))
        # Sort largest area first
        bboxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        return bboxes

    def _pad_bbox(
        self,
        bbox: tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        padding: float,
    ) -> tuple[int, int, int, int]:
        """Expand bbox by `padding` fraction, clamped to image dimensions."""
        x1, y1, x2, y2 = bbox
        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)
        return (
            max(0, x1 - pw),
            max(0, y1 - ph),
            min(img_w, x2 + pw),
            min(img_h, y2 + ph),
        )

    # ------------------------------------------------------------------
    # Single-image processing
    # ------------------------------------------------------------------

    def _process_single_image(
        self,
        img_rgb: np.ndarray,          # uint8 [H, W, 3]
        face_target: str,
        face_crop_padding: float,
        fallback_to_full_image: bool,
        detection_sensitivity: float,
        mask_expansion: int,
        edge_softness: int,
        mask_mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            full_mask: float32 [H, W] — the protection mask (1 = occluded, preserve)
            preview:   uint8  [H, W, 3] — debug overlay image
        """
        H, W = img_rgb.shape[:2]
        full_mask = np.zeros((H, W), dtype=np.float32)

        # 1. Detect faces
        bboxes = self._detect_faces(img_rgb)

        if not bboxes:
            _logger.warning("No faces detected in image.")
            if fallback_to_full_image:
                _logger.info("Fallback: treating full image as face region.")
                bboxes = [(0, 0, W, H)]
            else:
                # Return empty mask + annotated preview
                preview = _build_preview(img_rgb, full_mask, [])
                return full_mask, preview

        # 2. Select which faces to process
        if face_target == "Largest face only":
            bboxes_to_process = [bboxes[0]]
        else:
            bboxes_to_process = bboxes

        # 3. Per-face mask computation
        for bbox in bboxes_to_process:
            padded = self._pad_bbox(bbox, W, H, face_crop_padding)
            px1, py1, px2, py2 = padded
            crop_h, crop_w = py2 - py1, px2 - px1

            # Crop the face region
            face_crop = img_rgb[py1:py2, px1:px2]

            # Run occlusion inference at 256×256
            mask_256 = _compute_face_occluder_mask(
                face_crop_rgb=face_crop,
                detection_sensitivity=detection_sensitivity,
                mask_mode="Hard" if mask_mode.startswith("Hard") else "Soft",
                mask_expansion=mask_expansion,
                edge_softness=edge_softness,
                occluder_sess=self._get_occluder(),
                xseg_sess=self._get_xseg(),
            )  # float32 [256, 256]

            # Resize mask back to crop region size
            mask_crop = cv2.resize(
                mask_256, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
            )

            # Union into full-image mask
            full_mask[py1:py2, px1:px2] = np.maximum(
                full_mask[py1:py2, px1:px2], mask_crop
            )

        full_mask = np.clip(full_mask, 0.0, 1.0)

        # 4. Build debug preview
        preview = _build_preview(img_rgb, full_mask, bboxes_to_process)

        return full_mask, preview

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_occlusion_mask(
        self,
        input_image: torch.Tensor,
        face_target: str = "Largest face only",
        face_crop_padding: float = 0.15,
        fallback_to_full_image: bool = False,
        detection_sensitivity: float = 0.65,
        mask_expansion: int = 6,
        edge_softness: int = 4,
        mask_mode: str = "Soft (recommended)",
    ):
        """
        Accepts ComfyUI IMAGE tensors in two formats:
          - Single:  [1, H, W, 3]  float32
          - Batch:   [N, H, W, 3]  float32
          - List:    list of [1, H, W, 3] tensors  (from BatchLoadImages OUTPUT_IS_LIST)

        Returns: (IMAGE, MASK, PREVIEW) — each matching the batch dimension of the input.
        """
        occluder_sess = self._get_occluder()  # warm up before batch loop
        xseg_sess     = self._get_xseg()
        _                = occluder_sess, xseg_sess  # keep references alive

        # ── Normalise input to a list of individual uint8 [H,W,3] arrays ──
        if isinstance(input_image, list):
            # List of tensors from BatchLoadImages (OUTPUT_IS_LIST=True)
            frames_np = [_tensor_to_numpy_rgb(t) for t in input_image]
            was_list  = True
        elif isinstance(input_image, torch.Tensor):
            t = input_image
            if t.ndim == 3:
                t = t.unsqueeze(0)  # [H,W,3] → [1,H,W,3]
            # t is now [N, H, W, 3]
            frames_np = [_tensor_to_numpy_rgb(t[i].unsqueeze(0)) for i in range(t.shape[0])]
            was_list  = False
        else:
            raise TypeError(f"Unsupported input_image type: {type(input_image)}")

        out_images   = []
        out_masks    = []
        out_previews = []

        for img_rgb in frames_np:
            full_mask, preview = self._process_single_image(
                img_rgb=img_rgb,
                face_target=face_target,
                face_crop_padding=face_crop_padding,
                fallback_to_full_image=fallback_to_full_image,
                detection_sensitivity=detection_sensitivity,
                mask_expansion=mask_expansion,
                edge_softness=edge_softness,
                mask_mode=mask_mode,
            )
            out_images.append(_numpy_rgb_to_tensor(img_rgb))       # [1,H,W,3] passthrough
            out_masks.append(_mask_numpy_to_tensor(full_mask))     # [1,H,W]
            out_previews.append(_numpy_rgb_to_tensor(preview))     # [1,H,W,3]

        # ── Pack outputs back into the right format ──
        if len(out_images) == 1 and not was_list:
            # Single image — return [1,H,W,3] tensors directly
            image_out   = out_images[0]
            mask_out    = out_masks[0]
            preview_out = out_previews[0]
        else:
            # Batch — concatenate along dim 0
            # For MASK, ComfyUI expects [N, H, W] so we cat the [1,H,W] tensors
            image_out   = torch.cat(out_images,   dim=0)  # [N,H,W,3]
            mask_out    = torch.cat(out_masks,    dim=0)  # [N,H,W]
            preview_out = torch.cat(out_previews, dim=0)  # [N,H,W,3]

        return (image_out, mask_out, preview_out)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "OcclusionMask": OcclusionMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OcclusionMask": "Occlusion Mask (Face Protection for ReActor)",
}
