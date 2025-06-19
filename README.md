# ComfyUI OcclusionMask Custom Node

A powerful ComfyUI custom node for advanced face occlusion, segmentation, and masking, leveraging state-of-the-art face detection (insightface buffalo models) for robust and accurate results.

![Demo Screenshot](https://raw.githubusercontent.com/ialhabbal/OcclusionMask/main/media/Screenshot%202025-06-19%20205120.jpg)  
![Demo Screenshot](https://raw.githubusercontent.com/ialhabbal/OcclusionMask/main/media/Screenshot%202025-06-19%20205232.jpg)  
![Demo Screenshot](https://raw.githubusercontent.com/ialhabbal/OcclusionMask/main/media/Screenshot%202025-06-19%20205346.jpg)  
![Demo Screenshot](https://raw.githubusercontent.com/ialhabbal/OcclusionMask/main/media/Screenshot%202025-06-19%20212522.jpg)



## Features
- **Automatic Face Detection:** Uses insightface's FaceAnalysis API with buffalo models for highly accurate face localization.
- **Automatic Fallback:** If no face is detected, the node automatically processes the whole image.
- **Multiple Mask Types:** Choose between Occluder, XSeg, or Object-only masks for flexible workflows.
- **Mask Placement:** The generated mask is placed back onto the original image in the correct location and size.
- **Fine Mask Control:**
  - Adjustable mask threshold
  - Feather/blur radius
  - Directional mask growth/shrink (left, right, up, down)
  - Dilation and expansion iterations
- **ONNX Runtime Acceleration:** Fast inference using ONNX models with CUDA or CPU fallback.
- **Handles Multiple Image Types:** Accepts PIL Images, numpy arrays, or torch tensors as input, and robustly converts them for processing.
- **Batch Image Loading (Please check OcclusionMask_workflow and OcclusionMask_workflow_video):**
  - Load all images from a directory (supports .png, .jpg, .jpeg, .webp, .bmp, .gif)
  - Optionally process batches from video nodes or other sources
  - Option to use input images from other nodes for flexible workflow chaining
- **Outputs:**
  - Returns both the original image and the selected mask for further processing in ComfyUI workflows
- **Easy Integration:** Designed for seamless use in ComfyUI custom node pipelines with node export mappings for easy node registration.

## Requirements
- Python 3.8+
- ComfyUI (latest recommended)
- Windows (tested), Linux should work with minor path adjustments

## Pre-requisites
- Download and set up [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Download insightface buffalo models (see below)
- Place ONNX models for occlusion and XSeg in the `models` folder inside this node

## Dependencies
- `onnxruntime`
- `torch`
- `opencv-python`
- `numpy`
- `Pillow`
- `insightface`

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Installation
1. **Clone or copy this repository into your ComfyUI custom_nodes directory:**
   ```
   <ComfyUI root>/custom_nodes/OcclusionMask/
   ```
2. **Place ONNX models:**
   - `occluder.onnx` and `XSeg_model.onnx` in `OcclusionMask/models/`
   
   - Download Occluder.onnx from: https://huggingface.co/OwlMaster/AllFilesRope/blob/d783e61585b3d83a85c91ca8a3b299e8ade94d72/occluder.onnx
   - Download Xseg_model.onnx from: https://huggingface.co/OwlMaster/AllFilesRope/blob/d783e61585b3d83a85c91ca8a3b299e8ade94d72/XSeg_model.onnx

3. **Place insightface models:**
   - Download buffalo models (e.g., `buffalo_l`) from [insightface model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
   - Place them in a directory, e.g.,
     ```
     <ComfyUI root>/models/insightface/
     ```
   - The code sets `INSIGHTFACE_HOME` to this path automatically.
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Add the **Image Occlusion Node** to your ComfyUI workflow.
- Add the **Batch Image Loader Node** to process multiple images from a directory or from another node.
- Connect an input image and configure the mask type and parameters as desired.
- The node will output the original image and the selected mask.

### Node Parameters
- **Mask Type:** Occluder, XSeg, or Object-only
- **Object Mask Threshold:** Float slider (0.0–1.0)
- **Feather/Blur Radius:** Integer slider (0–64 px)
- **Grow/Shrink (Left, Right, Up, Down):** Integer sliders (-64 to 64 px)
- **Dilation Radius:** Integer slider (0–32 px)
- **Expansion Iterations:** Float slider (0.0–10.0)

## Folder Structure
```
OcclusionMask/
├── face_helpers/
│   ├── __init__.py
│   └── face_masks.py
├── models/
│   ├── occluder.onnx
│   └── XSeg_model.onnx
├── requirements.txt
├── Occlusion.py
├── batch_comfyui_processor.py
└── README.md
```

## Troubleshooting
- If you see import errors for `onnxruntime`, `torch`, `cv2`, or `insightface`, ensure your Python environment is activated and all dependencies are installed.
- If face detection fails, check that your buffalo models are present in the correct directory and that `INSIGHTFACE_HOME` is set (the node sets this automatically).

## Note for ReActor Node Users

If you have already installed the [ReActor Custom Node] (https://github.com/Gourieff/ComfyUI-ReActor) for ComfyUI, many of the required Python packages (such as `insightface`, `onnxruntime`, `torch`, `opencv-python`, `numpy`, and `Pillow`) and the necessary face detection models are likely already installed and set up.

You may not need to reinstall these dependencies or download the models again. Simply ensure your environment is activated and proceed with the installation steps for this node.

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — The base UI and node framework.
- [insightface](https://github.com/deepinsight/insightface) — Face detection and recognition models.
- [DeepFaceLab](https://github.com/iperov/DeepFaceLab) — For XSeg and occluder models.
- [VisoMaster](https://github.com/visomaster/VisoMaster) - "Helper" functions adaptation. 
- [ReActor Custom Node](https://github.com/Gourieff/ComfyUI-ReActor) - Inspiration for this custom node.

## License
MIT License

Developed by [ialhabbal]

## Contributing

Contributions, suggestions, and improvements from the community are very welcome!

- Feel free to open issues for bugs, questions, or feature requests.
- Pull requests for new features, bug fixes, or documentation improvements are encouraged.
- Please follow standard open-source etiquette and provide clear commit messages.
