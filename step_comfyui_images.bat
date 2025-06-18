@echo off
REM This batch file runs a Python script to automate stepping through images in ComfyUI
REM Make sure you have Python installed and requests library available

set /p WORKFLOW_FILE="Enter the workflow JSON filename (e.g. workflow.json): "
set /p FOLDER_PATH="Enter the full path to your image folder: "

REM You may need to adjust the path to python.exe if not in PATH
python step_comfyui_images.py "%WORKFLOW_FILE%" "%FOLDER_PATH%"
pause
