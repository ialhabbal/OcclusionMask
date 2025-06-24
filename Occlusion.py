import os
os.environ["INSIGHTFACE_HOME"] = r"L:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\insightface"
import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
import cv2  # Add OpenCV for dilation
from .face_helpers.face_masks import FaceMasks
# --- Use insightface FaceAnalysis for detection (buffalo models) ---
from insightface.app import FaceAnalysis

class ImageOcclusion:
    def __init__(self):
        # Set model paths
        base_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.model_paths = {
            'Occlusion Mask': os.path.join(base_dir, 'occluder.onnx'),
            'DFL XSeg Mask': os.path.join(base_dir, 'XSeg_model.onnx'),
        }
        # Try CUDA if available, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.occluder_sess = ort.InferenceSession(self.model_paths['Occlusion Mask'], providers=providers)
        self.xseg_sess = ort.InferenceSession(self.model_paths['DFL XSeg Mask'], providers=providers)
        print("[DEBUG] Occluder model inputs:", self.occluder_sess.get_inputs())
        print("[DEBUG] Occluder model outputs:", self.occluder_sess.get_outputs())
        print("[DEBUG] XSeg model inputs:", self.xseg_sess.get_inputs())
        print("[DEBUG] XSeg model outputs:", self.xseg_sess.get_outputs())
        self.facemasks = FaceMasks(device='cpu', model_occluder=self.occluder_sess, model_xseg=self.xseg_sess)
        # --- Load insightface FaceAnalysis with buffalo models ---
        self.face_detector = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        self.face_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"forceInput": True, "label": "Photo 1 (original)"}),
                "mask_type": (["Occluder", "XSeg", "Object-only"], {"default": "Object-only", "label": "Mask Type"}),
                "object_mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "label": "Object Mask Threshold"}),
                "feather_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "display": "slider", "label": "Feather/Blur Radius (px)"}),
                "grow_left": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "display": "slider", "label": "Grow Left (px)"}),
                "grow_right": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "display": "slider", "label": "Grow Right (px)"}),
                "grow_up": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "display": "slider", "label": "Grow Up (px)"}),
                "grow_down": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "display": "slider", "label": "Grow Down (px)"}),
                "dilation_radius": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1, "display": "slider", "label": "Dilation Radius (px)"}),
                "expansion_iterations": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "slider", "label": "Expansion Iterations"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("OriginalImage", "SelectedMask")
    FUNCTION = "apply_mask"
    CATEGORY = "image/mynode2"

    def _to_pil(self, img):
        # Convert tensor/array to PIL Image if needed
        if isinstance(img, Image.Image):
            return img
        elif hasattr(img, 'cpu') and hasattr(img, 'numpy'):
            arr = img.cpu().numpy()
        else:
            arr = np.array(img)
        # Remove all leading singleton dimensions
        arr = np.squeeze(arr)
        # If shape is (H, W), make it (H, W, 1)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        # If shape is (C, H, W), transpose to (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in [1, 3]:
            arr = np.transpose(arr, (1, 2, 0))
        # If shape is (H, W, C) and dtype is not uint8, scale
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        # If still not (H, W, C), try to fix
        if arr.ndim != 3 or arr.shape[2] not in [1, 3]:
            raise ValueError(f"Cannot convert array of shape {arr.shape} to image")
        return Image.fromarray(arr)

    def _detect_face_bbox(self, pil_img):
        # Use insightface FaceAnalysis to detect face bounding box
        img_np = np.array(pil_img.convert('RGB'))
        faces = self.face_detector.get(img_np)
        if not faces:
            return None  # No face detected
        # Get the largest face
        max_area = 0
        best_bbox = None
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_bbox = (x1, y1, x2, y2)
        return best_bbox

    def apply_mask(self, input_image, mask_type="Object-only", object_mask_threshold=0.5, feather_radius=0, grow_left=0, grow_right=0, grow_up=0, grow_down=0, dilation_radius=0, expansion_iterations=1.0):
        orig_pil = self._to_pil(input_image)
        orig_size = orig_pil.size
        # --- Face detection ---
        bbox = self._detect_face_bbox(orig_pil)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Clamp bbox to image size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_size[0], x2), min(orig_size[1], y2)
            face_crop = orig_pil.crop((x1, y1, x2, y2)).resize((256, 256), Image.BILINEAR)
            crop_box = (x1, y1, x2, y2)
        else:
            # Fallback to whole image
            face_crop = orig_pil.resize((256, 256), Image.BILINEAR)
            crop_box = (0, 0, orig_size[0], orig_size[1])
        img_np = np.array(face_crop).astype(np.float32)
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).to('cpu')
        # Occluder mask
        occluder_mask_tensor = self.facemasks.apply_occlusion(img_tensor, amount=0)
        occluder_mask_np = occluder_mask_tensor.squeeze().cpu().numpy()
        occluder_mask_img = Image.fromarray((occluder_mask_np * 255).astype(np.uint8), mode='L').resize((crop_box[2]-crop_box[0], crop_box[3]-crop_box[1]), Image.BILINEAR)
        occluder_mask_arr = np.array(occluder_mask_img).astype(np.float32) / 255.0
        # XSeg mask
        xseg_mask_tensor = self.facemasks.apply_dfl_xseg(img_tensor, amount=0)
        xseg_mask_np = xseg_mask_tensor.squeeze().cpu().numpy()
        xseg_mask_img = Image.fromarray((xseg_mask_np * 255).astype(np.uint8), mode='L').resize((crop_box[2]-crop_box[0], crop_box[3]-crop_box[1]), Image.BILINEAR)
        xseg_mask_arr = np.array(xseg_mask_img).astype(np.float32) / 255.0
        # Object-only mask: occluder - xseg, clamp to [0,1]
        object_only_arr = np.clip(occluder_mask_arr - xseg_mask_arr, 0, 1)
        # Select mask
        if mask_type == "Occluder":
            mask_arr = occluder_mask_arr
        elif mask_type == "XSeg":
            mask_arr = xseg_mask_arr
        else:
            mask_arr = object_only_arr
        # Feather/blur the mask before thresholding
        if feather_radius > 0:
            mask_arr = cv2.GaussianBlur(mask_arr, (feather_radius*2+1, feather_radius*2+1), 0)
        # Apply threshold: everything above threshold is 1, else 0
        mask_arr = (mask_arr >= object_mask_threshold).astype(np.float32)
        # Directional grow/shrink: shift and combine or intersect mask in each direction
        h, w = mask_arr.shape
        grown_mask = mask_arr.copy()
        if grow_left > 0:
            grown_mask = np.maximum(grown_mask, np.pad(mask_arr[:, :-grow_left], ((0,0),(grow_left,0)), mode='constant'))
        elif grow_left < 0:
            grown_mask = grown_mask.copy()
            grown_mask[:, :abs(grow_left)] = 0
        if grow_right > 0:
            grown_mask = np.maximum(grown_mask, np.pad(mask_arr[:, grow_right:], ((0,0),(0,grow_right)), mode='constant'))
        elif grow_right < 0:
            grown_mask = grown_mask.copy()
            grown_mask[:, -abs(grow_right):] = 0
        if grow_up > 0:
            grown_mask = np.maximum(grown_mask, np.pad(mask_arr[grow_up:, :], ((0,grow_up),(0,0)), mode='constant'))
        elif grow_up < 0:
            grown_mask = grown_mask.copy()
            grown_mask[:abs(grow_up), :] = 0
        if grow_down > 0:
            grown_mask = np.maximum(grown_mask, np.pad(mask_arr[:-grow_down, :], ((grow_down,0),(0,0)), mode='constant'))
        elif grow_down < 0:
            grown_mask = grown_mask.copy()
            grown_mask[-abs(grow_down):, :] = 0
        mask_arr = grown_mask
        # Dilation: expand the mask by dilation_radius pixels, for expansion_iterations times
        if dilation_radius > 0 and expansion_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_radius+1, 2*dilation_radius+1))
            int_iter = int(np.floor(expansion_iterations))
            frac_iter = expansion_iterations - int_iter
            mask = mask_arr.copy()
            if int_iter > 0:
                mask = cv2.dilate(mask, kernel, iterations=int_iter)
            if frac_iter > 0:
                mask1 = mask
                mask2 = cv2.dilate(mask, kernel, iterations=1)
                mask = (1 - frac_iter) * mask1 + frac_iter * mask2
            mask_arr = np.clip(mask, 0, 1)
        # --- Place mask back on original image ---
        full_mask = np.zeros((orig_pil.height, orig_pil.width), dtype=np.float32)
        x1, y1, x2, y2 = crop_box
        mask_resized = cv2.resize(mask_arr, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
        full_mask[y1:y2, x1:x2] = mask_resized
        mask_out = torch.from_numpy(full_mask[None, ...].copy()).float()  # [1,H,W]
        return (input_image, mask_out)

# Node export mappings
NODE_CLASS_MAPPINGS = {"ImageOcclusion": ImageOcclusion}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageOcclusion": "Image Occlusion Node"}
