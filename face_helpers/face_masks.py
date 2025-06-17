import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2

class FaceMasks:
    def __init__(self, device='cpu', model_occluder=None, model_xseg=None):
        self.device = device
        self.model_occluder = model_occluder
        self.model_xseg = model_xseg

    def apply_occlusion(self, img, amount):
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.device).contiguous()

        self.run_occluder(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount >0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)
            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)
            outpred = torch.squeeze(outpred)
        if amount <0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)
            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_occluder(self, image, output):
        if self.model_occluder is None:
            raise RuntimeError('Occluder model not loaded!')
        img_np = image.detach().cpu().numpy().astype(np.float32)
        # Ensure batch dimension
        if img_np.shape == (3, 256, 256):
            img_np = np.expand_dims(img_np, 0)
        print("[DEBUG] Occluder ONNX input name:", self.model_occluder.get_inputs()[0].name)
        print("[DEBUG] Occluder ONNX input shape:", self.model_occluder.get_inputs()[0].shape)
        print("[DEBUG] Occluder input numpy shape:", img_np.shape)
        ort_inputs = {self.model_occluder.get_inputs()[0].name: img_np}
        mask = self.model_occluder.run(None, ort_inputs)[0]
        mask_tensor = torch.from_numpy(mask).to(self.device)
        output.copy_(mask_tensor.reshape_as(output))

    def apply_dfl_xseg(self, img, amount):
        img = img.type(torch.float32)
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.device).contiguous()
        self.run_dfl_xseg(img, outpred)
        outpred = torch.clamp(outpred, min=0.0, max=1.0)
        outpred[outpred < 0.1] = 0
        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)
        if amount > 0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)
            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)
            outpred = torch.squeeze(outpred)
        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)
            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_dfl_xseg(self, image, output):
        if self.model_xseg is None:
            raise RuntimeError('XSeg model not loaded!')
        img_np = image.detach().cpu().numpy().astype(np.float32)
        if img_np.shape == (3, 256, 256):
            img_np = np.expand_dims(img_np, 0)
        print("[DEBUG] XSeg ONNX input name:", self.model_xseg.get_inputs()[0].name)
        print("[DEBUG] XSeg ONNX input shape:", self.model_xseg.get_inputs()[0].shape)
        print("[DEBUG] XSeg input numpy shape:", img_np.shape)
        ort_inputs = {self.model_xseg.get_inputs()[0].name: img_np}
        mask = self.model_xseg.run(None, ort_inputs)[0]
        mask_tensor = torch.from_numpy(mask).to(self.device)
        output.copy_(mask_tensor.reshape_as(output))
