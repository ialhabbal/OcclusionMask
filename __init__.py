# __init__.py

# Import node class mappings
from .Occlusion import NODE_CLASS_MAPPINGS as OCCLUSION_NODES
from .Occlusion import NODE_DISPLAY_NAME_MAPPINGS as OCCLUSION_DISPLAY

from .batch_comfyui_processor import NODE_CLASS_MAPPINGS as BATCH_NODES
from .batch_comfyui_processor import NODE_DISPLAY_NAME_MAPPINGS as BATCH_DISPLAY

# Merge all class mappings
NODE_CLASS_MAPPINGS = {
    **OCCLUSION_NODES,
    **BATCH_NODES,
}

# Merge all display name mappings (optional but good practice)
NODE_DISPLAY_NAME_MAPPINGS = {
    **OCCLUSION_DISPLAY,
    **BATCH_DISPLAY,
}
