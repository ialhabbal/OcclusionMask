import os
from PIL import Image
import numpy as np
import torch

class BatchLoadImages:
    """
    Loads all images from a folder or accepts a batch of images (e.g., from a video node) and outputs a batch tensor suitable for ComfyUI workflows.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {"label": "Image Directory", "multiline": False, "placeholder": "Image Directory"}),
                "subdirectories": (["true", "false"], {"default": "false"}),
            },
            "optional": {
                "use_input_images": ("BOOLEAN", {"default": False, "label": "Use Input Images (from node)"}),
                "input_images": ("IMAGE", {"forceInput": False, "label": "Input Batch (optional)"}),
            }
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'batch'
    CATEGORY = 'Image/Batch'
    OUTPUT_IS_LIST = (True, )

    def batch(self, image_directory, subdirectories, use_input_images=False, input_images=None):
        # Use input_images only if toggle is enabled and input_images is provided
        if use_input_images and input_images is not None:
            imgs = input_images
            # If input is a tensor batch [N, H, W, 3], split into list of [1, H, W, 3]
            if isinstance(imgs, torch.Tensor) and imgs.ndim == 4 and imgs.shape[-1] == 3:
                imgs = [img.unsqueeze(0) for img in imgs]
            elif not isinstance(imgs, list):
                imgs = [imgs]
            tensors = []
            for img in imgs:
                if isinstance(img, Image.Image):
                    img = img.convert('RGB')
                    arr = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).unsqueeze(0)
                elif isinstance(img, np.ndarray):
                    arr = img.astype(np.float32)
                    if arr.max() > 1.1:
                        arr = arr / 255.0
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, -1)
                    if arr.shape[-1] == 1:
                        arr = np.repeat(arr, 3, axis=-1)
                    tensor = torch.from_numpy(arr).unsqueeze(0)
                elif isinstance(img, torch.Tensor):
                    tensor = img.float()
                    if tensor.max() > 1.1:
                        tensor = tensor / 255.0
                    if tensor.ndim == 2:
                        tensor = tensor.unsqueeze(-1)
                    if tensor.shape[-1] == 1:
                        tensor = tensor.repeat(1, 1, 3)
                    if tensor.ndim == 3 and tensor.shape[-1] == 3:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[-1] == 3:
                        pass
                    else:
                        raise RuntimeError(f"[ERROR] Tensor is not RGB: got shape {tensor.shape}")
                else:
                    raise RuntimeError(f"Unsupported input type: {type(img)}")
                if tensor.shape[-1] != 3:
                    raise RuntimeError(f"[ERROR] Output is not [1, H, W, 3]: got {tensor.shape}")
                tensors.append(tensor)
            return (tensors,)
        # Otherwise, load from directory
        if not os.path.exists(image_directory):
            raise Exception(f"Image directory {image_directory} does not exist")
        images = []
        for file in os.listdir(image_directory):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.webp') or file.endswith('.bmp') or file.endswith('.gif'):
                img = Image.open(os.path.join(image_directory, file))
                img = img.convert('RGB')
                arr = np.array(img).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr).unsqueeze(0)
                images.append(tensor)
        return (images,)

NODE_CLASS_MAPPINGS = {"BatchLoadImages": BatchLoadImages}
NODE_DISPLAY_NAME_MAPPINGS = {"BatchLoadImages": "Loader for Batch Image Processing"}
