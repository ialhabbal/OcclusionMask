"""
FaceMasks helper — thin wrappers around the two ONNX sessions.

NOTE: This file is intentionally minimal. All mask-combination logic
now lives in Occlusion.py where it is easier to follow and test.
These methods are kept for potential reuse by other nodes.
"""

import numpy as np
import torch
import logging

_logger = logging.getLogger(__name__)


class FaceMasks:
    """
    Thin wrapper that keeps the two ONNX sessions and exposes clean
    numpy-in / numpy-out helpers.

    Both models expect a 256×256 RGB image and produce a 256×256 mask.
    """

    def __init__(self, device: str = "cpu", model_occluder=None, model_xseg=None):
        self.device = device
        self.model_occluder = model_occluder
        self.model_xseg = model_xseg

    # ------------------------------------------------------------------
    # Occluder model
    # ------------------------------------------------------------------

    def run_occluder_np(self, img_chw_float01: np.ndarray) -> np.ndarray:
        """
        img_chw_float01: float32 [3, 256, 256] in range [0, 1]
        Returns: float32 [256, 256] binary mask — 1 = occluder pixel
        """
        if self.model_occluder is None:
            raise RuntimeError("Occluder model not loaded.")
        inp = img_chw_float01[None].astype(np.float32)  # [1, 3, 256, 256]
        out = self.model_occluder.run(
            None, {self.model_occluder.get_inputs()[0].name: inp}
        )[0]
        mask = out.squeeze().astype(np.float32)
        return (mask > 0).astype(np.float32)

    def run_occluder(self, image: torch.Tensor, output: torch.Tensor):
        """
        Legacy torch-tensor interface (kept for backwards compatibility).
        image:  float32 tensor [3, 256, 256] or [1, 3, 256, 256], values in [0, 1]
        output: pre-allocated float32 tensor [256, 256] — filled in place
        """
        img_np = image.detach().cpu().numpy().astype(np.float32)
        if img_np.ndim == 4:
            img_np = img_np[0]  # [3, 256, 256]
        mask = self.run_occluder_np(img_np)
        output.copy_(torch.from_numpy(mask).to(self.device))

    # ------------------------------------------------------------------
    # XSeg model
    # ------------------------------------------------------------------

    def run_xseg_np(self, img_chw_float01: np.ndarray) -> np.ndarray:
        """
        img_chw_float01: float32 [3, 256, 256] in range [0, 1]
        Returns: float32 [256, 256] face-skin mask — 1 = face skin, 0 = object/background
        (XSeg raw output is non-face/object; this method returns the INVERTED version.)
        """
        if self.model_xseg is None:
            raise RuntimeError("XSeg model not loaded.")
        inp = img_chw_float01[None].astype(np.float32)  # [1, 3, 256, 256]
        out = self.model_xseg.run(
            None, {self.model_xseg.get_inputs()[0].name: inp}
        )[0]
        mask = out.squeeze().astype(np.float32)
        mask = np.clip(mask, 0.0, 1.0)
        mask[mask < 0.1] = 0.0
        # Invert: raw XSeg high = non-face → return face skin map
        return 1.0 - mask

    def run_dfl_xseg(self, image: torch.Tensor, output: torch.Tensor):
        """
        Legacy torch-tensor interface (kept for backwards compatibility).
        image:  float32 tensor [3, 256, 256] or [1, 3, 256, 256], values in [0, 1]
        output: pre-allocated float32 tensor [256, 256] — filled in place
        NOTE: this fills `output` with the RAW (non-inverted) XSeg output,
        matching the original behaviour for callers that invert themselves.
        """
        if self.model_xseg is None:
            raise RuntimeError("XSeg model not loaded.")
        img_np = image.detach().cpu().numpy().astype(np.float32)
        if img_np.ndim == 4:
            img_np = img_np[0]
        inp = img_np[None]
        out = self.model_xseg.run(
            None, {self.model_xseg.get_inputs()[0].name: inp}
        )[0]
        mask = out.squeeze().astype(np.float32)
        mask = np.clip(mask, 0.0, 1.0)
        mask_t = torch.from_numpy(mask).to(self.device)
        output.copy_(mask_t.reshape_as(output))
