# ComfyUI OcclusionMask Custom Node

A powerful ComfyUI custom node for advanced face occlusion, segmentation, and masking, leveraging state-of-the-art face detection (insightface buffalo models) for robust and accurate results.

## Features
- **Automatic Face Detection:** Uses insightface's FaceAnalysis API with buffalo models for highly accurate face localization.
- **Multiple Mask Types:** Choose between Occluder, XSeg, or Object-only masks for flexible workflows.
- **Fine Mask Control:**
  - Adjustable mask threshold
  - Feather/blur radius
  - Directional mask growth/shrink (left, right, up, down)
  - Dilation and expansion iterations
- **ONNX Runtime Acceleration:** Fast inference using ONNX models with CUDA or CPU fallback.
- **Easy Integration:** Designed for seamless use in ComfyUI custom node pipelines.

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
└── README.md
```

## Troubleshooting
- If you see import errors for `onnxruntime`, `torch`, `cv2`, or `insightface`, ensure your Python environment is activated and all dependencies are installed.
- If face detection fails, check that your buffalo models are present in the correct directory and that `INSIGHTFACE_HOME` is set (the node sets this automatically).

## Note for ReActor Node Users

If you have already installed the [ReActor Custom Node] (https://github.com/Gourieff/ComfyUI-ReActor) for ComfyUI, many of the required Python packages (such as `insightface`, `onnxruntime`, `torch`, `opencv-python`, `numpy`, and `Pillow`) and the necessary face detection models are likely already installed and set up.

You may not need to reinstall these dependencies or download the models again. Simply ensure your environment is activated and proceed with the installation steps for this node.

## Batch Automation Script for ComfyUI Workflows

### New: Batch Image Processing Script

This repository now includes a Python script and a batch file for automating the processing of a folder of images through your ComfyUI workflow, one image at a time, with no manual intervention.

#### Included Files
- **step_comfyui_images.py** — The main Python automation script.
- **step_comfyui_images.bat** — A Windows batch file to launch the script with prompts for your workflow and image folder.

#### What the Script Does
- Prompts you for your exported ComfyUI workflow JSON (API format) and the folder containing your images.
- Loops through all images in the specified folder (supports PNG, JPG, JPEG, BMP, GIF, TIFF).
- For each image:
  - Updates the Load Image node in your workflow to point to the current image (using the full path).
  - Triggers the workflow via the ComfyUI API.
  - Waits for the workflow to finish before moving to the next image.
- Warns you if the folder is empty.
- Works with images in any folder, not just the default ComfyUI `input` folder.

#### Where to Put the Script and Batch File
- You can place `step_comfyui_images.py` and `step_comfyui_images.bat` anywhere on your system.
- For convenience, you may want to keep them on your Desktop or in your ComfyUI root directory.

#### Steps Before Using the Script
1. **Construct your workflow in ComfyUI** as you would like it to run for each image.
2. **Export your workflow as API JSON** (use the "Export as API" option in ComfyUI). Once the workflow is successfully exported as api, you can close it inside ComfyUI if you prefer, the script will process the exported workflow as if it was open in ComfyUI (ComfyUI should be running though).
3. **Place your images in a folder** of your choice (e.g., `L:\test`).
4. **Run the batch file** (`step_comfyui_images.bat`), enter the path to your workflow JSON and your image folder when prompted.

#### Additional Notes
- The script will process all images in the folder, one by one, and stop automatically when done.
- The images will be saved in the directory you set in the SaveImage node.
- If you want to use the default ComfyUI `input` folder, simply enter its path when prompted.
- The script is compatible with any workflow that uses the standard Load Image node.
- If you move the script, make sure Python and the `requests` library are installed and available in your PATH.

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
