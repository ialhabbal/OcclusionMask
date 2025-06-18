import requests
import time
import json
import sys
import os

COMFYUI_URL = "http://127.0.0.1:8188"

if len(sys.argv) < 3:
    print("Usage: python step_comfyui_images.py <workflow.json> <folder_path>")
    sys.exit(1)

workflow_file = sys.argv[1]
folder_path = sys.argv[2]

# Set your Load Image node id here (as a string, e.g. "7")
LOAD_IMAGE_NODE_ID = "7"  # Change this if your node id is different

image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
all_files = os.listdir(folder_path)
image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_exts]
image_files.sort()  # Optional: sort alphabetically

if not image_files:
    print(f"Warning: No image files found in folder '{folder_path}'. Script will exit after this message.")

def wait_for_prompt(prompt_id):
    while True:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if resp.status_code == 200:
            data = resp.json()
            # The response is a dict with the prompt_id as the key
            prompt_data = data.get(prompt_id)
            if prompt_data:
                # Check for error
                for k, v in prompt_data.items():
                    if isinstance(v, dict) and 'error' in v:
                        print(f"Error in workflow history: {v['error']}")
                        return 'error'
                # Check for completed or outputs
                status = prompt_data.get("status", {})
                outputs = prompt_data.get("outputs", {})
                if status.get("completed") or outputs:
                    print(f"Prompt {prompt_id} completed or outputs present.")
                    break
            else:
                print(f"Prompt id {prompt_id} not found in history response.")
        else:
            print(f"Failed to get history for {prompt_id}: {resp.status_code}")
        time.sleep(2)

for idx, image_file in enumerate(image_files):
    print(f"Processing image {idx+1} of {len(image_files)}: {image_file}")
    with open(workflow_file, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    # Update the Load Image node's input
    found = False
    for node_id, node in workflow.items():
        if node.get("class_type", "") == "LoadImage":
            print(f"Updating node {node_id} with image: {image_file}")
            node["inputs"]["image"] = os.path.join(folder_path, image_file)  # Use full path
            found = True
            break
    if not found:
        print(f"Could not find Load Image node in workflow. Skipping {image_file}.")
        continue
    response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    try:
        result = response.json()
        if 'error' in result:
            print(f"Error from ComfyUI: {result['error']}")
            continue
        prompt_id = result.get("prompt_id")
        if prompt_id:
            print(f"Triggered workflow for prompt {prompt_id}. Waiting for completion...")
            wf_result = wait_for_prompt(prompt_id)
            if wf_result == 'error':
                print("Workflow error detected, skipping to next image.")
                continue
        else:
            print("No prompt_id returned, skipping to next image.")
            continue
    except Exception as e:
        print(f"Error parsing response: {e}")
        continue
print("All images processed.")