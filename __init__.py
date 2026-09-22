"""Node registration.

The upstream module is imported best-effort. `nodes.py` uses absolute imports
for its own siblings (``from condtions import ...``, ``from tools.visualize
import ...``), which only resolve when the package directory happens to be on
sys.path -- true when run as a script, false when ComfyUI imports it as a
package. Rather than let that take the whole pack down at load time, the
directory is added to sys.path and the import is guarded.

The deployment-shaped nodes in `everwhere_nodes` are what the API actually
calls, and they must register whether or not the upstream CLI wrapper loads.
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- upstream CLI-shaped nodes: optional -----------------------------------
try:
    from .nodes import (
        GameVideoGenerator,
        LoadDiTModel,
        LoadGameImage,
        LoadMouseIcon,
        LoadTextEncoderModel,
        LoadVAEModel,
        MatrixGameOutput,
    )

    NODE_CLASS_MAPPINGS.update({
        "LoadDiTModel": LoadDiTModel,
        "LoadVAEModel": LoadVAEModel,
        "LoadTextEncoderModel": LoadTextEncoderModel,
        "LoadGameImage": LoadGameImage,
        "LoadMouseIcon": LoadMouseIcon,
        "GameVideoGenerator": GameVideoGenerator,
        "MatrixGameOutput": MatrixGameOutput,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "LoadDiTModel": "Load DiT Model",
        "LoadVAEModel": "Load VAE Model",
        "LoadTextEncoderModel": "Load TextEncoder Model",
        "LoadGameImage": "Load Game Image",
        "LoadMouseIcon": "Load Mouse Icon",
        "GameVideoGenerator": "Game Video Generator",
        "MatrixGameOutput": "MatrixGame Output",
    })
except Exception:
    print("[Matrix-Game] upstream CLI nodes unavailable, continuing:")
    traceback.print_exc()

# --- deployment-shaped nodes: required -------------------------------------
from .everwhere_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as _EW_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _EW_NAMES,
)

NODE_CLASS_MAPPINGS.update(_EW_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_EW_NAMES)

# --- Matrix-Game 2.0 nodes: required ---------------------------------------
# A separate architecture from 1.0 (causal Wan2.1 DiT + action module), so a
# separate module. Its heavy imports are deferred to execution time, which is
# why this import is safe at load.
from .matrixgame2_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as _MG2_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MG2_NAMES,
)

NODE_CLASS_MAPPINGS.update(_MG2_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_MG2_NAMES)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
