"""Deployment-shaped nodes for Matrix-Game 2.0 action-conditioned generation.

Matrix-Game 2.0 is a different architecture from 1.0: a causal Wan2.1 DiT
(``CausalWanModel``, 30 blocks, dim 1536) with an ``ActionModule`` in the first
N blocks, a Wan2.1 VAE for encoding, a block-causal Wan VAE decoder with a
rolling feature cache, and an open-clip image encoder for the visual context.
There is no text encoder and no HunyuanVideo VAE in this path, so none of the
1.0 loaders apply.

What upstream ships and why it cannot be served as-is:

  * ``inference.py`` writes MP4s with ``process_video`` and returns nothing, so
    a deployment job would produce zero output assets. These nodes return an
    ``IMAGE`` batch instead -- ``[T, H, W, C]``, float 0..1, on CPU -- which is
    what makes SaveWEBM / CreateVideo emit real assets.
  * the action track is a randomised benchmark (``Bench_actions_*``), not an
    input. Here keyboard and mouse arrive as JSON and become per-frame tensors.
  * ``inference_streaming.py`` calls ``input()`` between blocks. That is
    interactive by construction and is not wrapped here; see mg2/NOTES.md.

The action space differs per checkpoint, so it is read from the checkpoint's
own ``config.json`` (``action_config.keyboard_dim_in`` / ``enable_mouse``) and
cross-checked against the declared mode:

    universal   keyboard [T, 4] (forward, back, left, right)   mouse [T, 2]
    gta_drive   keyboard [T, 2] (forward, back)                mouse [T, 2]
    templerun   keyboard [T, 7] (nomove, jump, slide, turnleft,
                                 turnright, leftside, rightside)  no mouse

``T`` here is the *pixel* frame count ``(num_output_frames - 1) * 4 + 1``; the
action track is per pixel frame, while the latent track is 4x shorter.

Model weights load from local paths only. No ``from_pretrained`` hub call, no
``CUDA_VISIBLE_DEVICES`` mutation, no ``os.chdir``, no relative-path writes.
``flash_attn`` is shimmed onto SDPA when absent instead of imported hard.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import threading
import types

# Upstream's own source tree is vendored under mg2/Matrix-Game-2 and imported
# verbatim so it stays diffable against upstream. Its modules use absolute
# imports (``from utils.wan_wrapper import ...``), which collide with names a
# ComfyUI process already owns -- ComfyUI has its own top-level ``utils``
# package -- so the import is set up explicitly rather than by prepending to
# sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDORED_SRC = os.path.join(_HERE, "mg2", "Matrix-Game-2")

# Package names the vendored tree imports by absolute name, parents first.
# ``wan`` and ``wan.modules`` are claimed as namespace packages so their
# __init__ files never run: wan/__init__.py pulls WanI2V/WanT2V and
# wan/modules/__init__.py pulls wan/modules/t5.py, which calls
# torch.cuda.current_device() in a *class body* and therefore demands a GPU at
# import time. None of that is on this path.
_MG2_PACKAGES = ("wan", "wan.modules", "utils", "pipeline", "demo_utils")

MODES = ("universal", "gta_drive", "templerun")

# Per-mode facts that upstream hardcodes. keyboard_dim is cross-checked against
# the checkpoint's config.json rather than trusted.
MODE_KEYBOARD_DIM = {"universal": 4, "gta_drive": 2, "templerun": 7}
MODE_HAS_MOUSE = {"universal": True, "gta_drive": True, "templerun": False}
MODE_CONFIG_YAML = {
    "universal": "configs/inference_yaml/inference_universal.yaml",
    "gta_drive": "configs/inference_yaml/inference_gta_drive.yaml",
    "templerun": "configs/inference_yaml/inference_templerun.yaml",
}
MODE_MODEL_CONFIG = {
    "universal": "configs/distilled_model/universal",
    "gta_drive": "configs/distilled_model/gta_drive",
    "templerun": "configs/distilled_model/templerun",
}
KEYBOARD_KEYS = {
    "universal": ["forward", "back", "left", "right"],
    "gta_drive": ["forward", "back"],
    "templerun": ["nomove", "jump", "slide", "turnleft", "turnright",
                  "leftside", "rightside"],
}

# Fixed by the architecture: demo_utils.constant.ZERO_VAE_CACHE is built for a
# 44x80 latent, and CausalInferencePipeline.frame_seq_length == 880 == 44*80/4.
LATENT_H, LATENT_W = 44, 80
PIXEL_H, PIXEL_W = LATENT_H * 8, LATENT_W * 8  # 352 x 640
VAE_TIME_COMPRESSION = 4

_IMPORT_LOCK = threading.Lock()
_MG2 = None  # cached handles into the vendored tree


# --------------------------------------------------------------------------- #
# import plumbing
# --------------------------------------------------------------------------- #
def _install_flash_attn_shim():
    """Register an SDPA-backed ``flash_attn`` when the real wheel is missing.

    ``wan/modules/action_module.py`` imports ``flash_attn_func`` at module
    scope, and flash-attn needs a CUDA compiler at install time, so it is not
    in the deployment image. SDPA reaches the same FlashAttention kernels on
    Ampere and newer. The shim is registered in ``sys.modules`` so the vendored
    source stays byte-identical to upstream.

    flash_attn_func takes and returns [B, L, H, D]; SDPA wants [B, H, L, D].
    """
    if "flash_attn" in sys.modules:
        return
    if importlib.util.find_spec("flash_attn") is not None:
        return  # real wheel present; let the normal import win

    import torch
    import torch.nn.functional as F

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                        causal=False, **_ignored):
        qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            qt, kt, vt,
            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.transpose(1, 2)

    def flash_attn_varlen_func(*_a, **_k):
        # Deliberately a tripwire, not a fallback. Both vendored attention.py
        # files detect this shim (SDPA_SHIM / __version__ "0.0.0"), report
        # FLASH_ATTN_2_AVAILABLE = False, and take their own SDPA path, so
        # nothing should call this. Reimplementing varlen packing here would
        # silently paper over a future caller that this shim gets wrong --
        # and wrong attention is wrong video, which is worse than a crash.
        raise RuntimeError(
            "flash_attn_varlen_func reached the SDPA shim. Both "
            "wan/modules/attention.py and wan/vae/wanx_vae_src/attention.py "
            "are patched to route around it; a new caller needs the same "
            "treatment, or a real varlen implementation honouring "
            "cu_seqlens_q/cu_seqlens_k and the packed return layout."
        )

    def flash_attn_qkvpacked_func(*_a, **_k):
        # Not an alias for flash_attn_func: the real one takes a single packed
        # [B, L, 3, H, D] tensor, so aliasing it would be a wrong-signature
        # trap. Nothing in this tree calls it (`grep -rn qkvpacked` is empty).
        raise RuntimeError(
            "flash_attn_qkvpacked_func is not implemented by the SDPA shim; "
            "unpack qkv and call flash_attn_func instead."
        )

    mod = types.ModuleType("flash_attn")
    mod.__doc__ = "SDPA-backed stand-in installed by ComfyUI-Matrix-Game."
    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    mod.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    # __spec__ is not optional. diffusers and transformers both probe with
    # importlib.util.find_spec("flash_attn") at import time, and find_spec
    # raises ValueError for a module that is in sys.modules with __spec__ None.
    # Version 0.0.0 keeps their ">= 2.1.0" FlashAttention-2 gates closed, so
    # they route around flash-attn as they would if it were absent.
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    mod.__version__ = "0.0.0"
    mod.SDPA_SHIM = True
    sys.modules["flash_attn"] = mod


def _claim_package(name: str, directory: str):
    """Make ``directory`` searchable as the top-level package ``name``.

    ``utils`` and ``pipeline`` are names other code in the process may already
    own (ComfyUI ships ``utils/``). Extending the existing package's
    ``__path__`` keeps that package working -- none of the submodule names
    overlap -- and avoids touching ``sys.path``, where a later insert could
    shadow either side.
    """
    mod = sys.modules.get(name)
    if mod is None:
        # Keep any other location for this name searchable too, so a package
        # that has not been imported yet still resolves after we claim it.
        other = []
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            spec = None
        if spec is not None and spec.submodule_search_locations:
            other = [p for p in spec.submodule_search_locations
                     if os.path.abspath(p) != os.path.abspath(directory)]
        mod = types.ModuleType(name)
        mod.__path__ = other + [directory]
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name, None, is_package=True)
        mod.__spec__.submodule_search_locations = mod.__path__
        sys.modules[name] = mod
        # A synthesised submodule is not bound on its parent automatically.
        parent_name, _, leaf = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, leaf, mod)
        return

    path = list(getattr(mod, "__path__", []) or [])
    if directory not in path:
        path.append(directory)
        mod.__path__ = path
        spec = getattr(mod, "__spec__", None)
        if spec is not None and spec.submodule_search_locations is not None:
            try:
                spec.submodule_search_locations = path
            except Exception:
                pass


def _stub_visualize_if_no_cv2():
    """``pipeline.causal_inference`` imports ``utils.visualize`` for its MP4
    overlay helpers. Those are exactly what these nodes replace, so a missing
    opencv must not take the whole pipeline down."""
    if importlib.util.find_spec("cv2") is not None:
        return
    if "utils.visualize" in sys.modules:
        return

    def _unavailable(*_a, **_k):
        raise RuntimeError(
            "utils.visualize needs opencv-python; these nodes return an IMAGE "
            "batch instead of writing video files, so it is not installed."
        )

    stub = types.ModuleType("utils.visualize")
    stub.process_video = _unavailable
    stub.parse_config = _unavailable
    sys.modules["utils.visualize"] = stub
    parent = sys.modules.get("utils")
    if parent is not None:
        parent.visualize = stub


def source_root_candidates() -> list:
    """Places the vendored/official Matrix-Game-2 source may live, in order."""
    cands = [
        os.environ.get("MATRIXGAME2_SRC"),
        _VENDORED_SRC,
        os.path.join(_HERE, "Matrix-Game-2"),
    ]
    comfy = os.path.dirname(os.path.dirname(_HERE))  # .../ComfyUI
    cands += [
        os.path.join(comfy, "Matrix-Game", "Matrix-Game-2"),
        os.path.join(os.path.dirname(comfy), "Matrix-Game", "Matrix-Game-2"),
        "/workspace/dpbuild/Matrix-Game/Matrix-Game-2",
    ]
    return [c for c in cands if c]


def _resolve_source_root(explicit: str = "") -> str:
    cands = [explicit] if explicit else source_root_candidates()
    for c in cands:
        c = os.path.abspath(os.path.expanduser(c))
        if os.path.isdir(os.path.join(c, "wan", "modules")) and \
                os.path.isfile(os.path.join(c, "pipeline", "causal_inference.py")):
            return c
    raise FileNotFoundError(
        "Matrix-Game 2.0 source tree not found. Looked in: "
        + ", ".join(cands)
        + ". Set the source_root input or the MATRIXGAME2_SRC env var to a "
          "directory containing wan/, pipeline/, utils/ and demo_utils/."
    )


def _load_mg2(source_root: str = ""):
    """Import the pieces of the upstream tree these nodes drive. Cached."""
    global _MG2
    with _IMPORT_LOCK:
        if _MG2 is not None:
            return _MG2

        root = _resolve_source_root(source_root)
        _install_flash_attn_shim()
        for name in _MG2_PACKAGES:
            directory = os.path.join(root, *name.split("."))
            if os.path.isdir(directory):
                _claim_package(name, directory)
        _stub_visualize_if_no_cv2()

        # Imported by module path, not via ``pipeline/__init__.py`` or
        # ``wan/__init__.py``: the latter pulls WanI2V/WanT2V/T5/prompt_extend,
        # none of which this path uses.
        causal_inference = importlib.import_module("pipeline.causal_inference")
        wan_wrapper = importlib.import_module("utils.wan_wrapper")
        vae_block3 = importlib.import_module("demo_utils.vae_block3")
        wanx_vae = importlib.import_module("wan.vae.wanx_vae")
        causal_model = importlib.import_module("wan.modules.causal_model")

        _MG2 = {
            "root": root,
            "CausalInferencePipeline": causal_inference.CausalInferencePipeline,
            "WanDiffusionWrapper": wan_wrapper.WanDiffusionWrapper,
            "VAEDecoderWrapper": vae_block3.VAEDecoderWrapper,
            "get_wanx_vae_wrapper": wanx_vae.get_wanx_vae_wrapper,
            "CausalWanModel": causal_model.CausalWanModel,
        }
        return _MG2


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def _folder_paths():
    try:
        import folder_paths
        return folder_paths
    except Exception:
        return None


def _resolve_path(path: str, what: str) -> str:
    """Resolve a user-supplied path against ComfyUI's models tree.

    A deployment stages weights at paths like ``models/matrixgame2/...``, so
    both ``matrixgame2/x`` (relative to the models dir) and
    ``models/matrixgame2/x`` (relative to the ComfyUI root) are accepted, as
    are absolute paths. Nothing is ever written, and the cwd is never changed.
    """
    path = os.path.expanduser((path or "").strip())
    if not path:
        raise ValueError(f"{what} is required")
    cands = []
    if os.path.isabs(path):
        cands.append(path)
    else:
        fp = _folder_paths()
        if fp is not None:
            models = getattr(fp, "models_dir", None)
            base = getattr(fp, "base_path", None)
            if models:
                cands.append(os.path.join(models, path))
            if base:
                cands.append(os.path.join(base, path))
        cands.append(os.path.abspath(path))
        # ComfyUI's registered folder categories. A deployment may stage
        # weights on a network volume that is registered as a folder type
        # rather than living under models_dir, so ask the registry too.
        if fp is not None:
            head = path.replace("\\", "/").split("/")[0]
            rest = path.replace("\\", "/").split("/")[1:]
            try:
                for root in fp.get_folder_paths(head) or []:
                    cands.append(os.path.join(root, *rest) if rest else root)
            except Exception:
                pass
            for cat in ("diffusion_models", "checkpoints", "vae", "unet"):
                try:
                    for root in fp.get_folder_paths(cat) or []:
                        cands.append(os.path.join(root, path))
                        cands.append(os.path.join(root, os.path.basename(path)))
                except Exception:
                    pass

    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"{what} not found. Tried: " + ", ".join(cands)
        + " || " + _layout_report(os.path.basename(path))
    )


def _layout_report(basename: str) -> str:
    """Describe what IS on disk, so a single failed job reveals the real layout.

    Without shell access to a deployment, a not-found error that only says
    what it looked for costs a full build-release-deploy cycle to learn
    nothing. This reports what actually exists instead.
    """
    import glob

    fp = _folder_paths()
    bits = []
    if fp is not None:
        md = getattr(fp, "models_dir", None)
        bits.append(f"models_dir={md}")
        bits.append(f"base_path={getattr(fp, 'base_path', None)}")
        if md and os.path.isdir(md):
            try:
                bits.append("models_dir entries=" + ",".join(sorted(os.listdir(md))[:40]))
            except Exception as e:
                bits.append(f"models_dir unreadable: {e}")
        try:
            names = sorted(getattr(fp, "folder_names_and_paths", {}).keys())
            bits.append("registered categories=" + ",".join(names[:40]))
        except Exception:
            pass
    hits = []
    for root in ("/app/ComfyUI/models", "/app/models", "/models", "/workspace/models",
                 "/runpod-volume", "/app/ComfyUI"):
        if not os.path.isdir(root):
            continue
        try:
            hits += glob.glob(os.path.join(root, "**", basename), recursive=True)[:3]
        except Exception:
            pass
    bits.append("found-by-search=" + (",".join(hits[:5]) if hits else "NONE"))
    return " | ".join(bits)


def _read_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class _Cfg:
    """Attribute-access config object.

    ``CausalInferencePipeline`` reads ``args.denoising_step_list``,
    ``args.warp_denoising_step``, ``args.context_noise`` and
    ``getattr(args, "num_frame_per_block", 1)``. A plain namespace satisfies
    all of that without pulling in omegaconf.
    """

    def __init__(self, data: dict):
        self.__dict__.update(data)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# action sequences
# --------------------------------------------------------------------------- #
def _parse_action_json(raw: str, width: int, frames: int, name: str):
    """Parse an action track into a ``[frames, width]`` list of rows.

    Accepted forms, all JSON:

      * one row held for the whole clip -- ``[1, 0, 0, 0]``
      * an explicit per-frame list    -- ``[[1,0,0,0], [0,0,1,0], ...]``
      * run-length segments           -- ``[{"repeat": 120, "value": [1,0,0,0]},
                                            {"repeat": 60,  "value": [0,1,0,0]}]``

    The clip is 597 frames at the default length, so run-length segments are
    the only practical way to describe a multi-part action from an API call.
    Short tracks are padded by repeating the last row, which is what you want
    when an action simply continues; long tracks are truncated.
    """
    raw = (raw or "").strip()
    if not raw:
        rows = [[0.0] * width]
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} is not valid JSON: {e}") from e
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(f"{name} must be a non-empty JSON list")

        if all(isinstance(x, (int, float)) for x in parsed):
            rows = [list(parsed)]
        else:
            rows = []
            for i, entry in enumerate(parsed):
                if isinstance(entry, dict):
                    if "value" not in entry:
                        raise ValueError(
                            f"{name}[{i}] is an object without a 'value' key")
                    value = entry["value"]
                    repeat = int(entry.get("repeat", 1))
                    if repeat < 1:
                        raise ValueError(
                            f"{name}[{i}] has repeat={repeat}, must be >= 1")
                    if not isinstance(value, list):
                        raise ValueError(f"{name}[{i}]['value'] must be a list")
                    rows.extend([list(value)] * repeat)
                elif isinstance(entry, list):
                    rows.append(list(entry))
                else:
                    raise ValueError(
                        f"{name}[{i}] must be a list of numbers or an object "
                        f"with 'value'/'repeat', got {type(entry).__name__}")

    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"{name} row {i} has {len(row)} values, expected {width}")
        for v in row:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"{name} row {i} must contain only numbers")

    if len(rows) < frames:
        rows = rows + [rows[-1]] * (frames - len(rows))
    return [[float(v) for v in r] for r in rows[:frames]]


def pixel_frames(num_output_frames: int) -> int:
    """Action-track length for a given latent length."""
    return (num_output_frames - 1) * VAE_TIME_COMPRESSION + 1


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
class MatrixGame2PipelineLoader:
    """Build the Matrix-Game 2.0 pipeline once and cache it across jobs.

    Upstream constructs the DiT, both VAE halves and the CLIP image encoder on
    every run because it is a one-shot CLI. A deployment worker is warm and
    serves many jobs, so the build is cached on the resolved paths: the
    difference between a ~10s response and a ~2min one.

    Only one pipeline is kept resident -- each one is a 1.3B DiT plus a Wan VAE
    plus an open-clip ViT-H, and the causal KV caches on top are another ~1.6GB
    -- so switching mode evicts the previous one.
    """

    _CACHE: dict = {}
    _CACHE_MAXSIZE = 1
    _LOCK = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(MODES), {
                    "default": "universal",
                    "tooltip": "Selects the action space and the inference "
                               "yaml. Must match the checkpoint.",
                }),
                "checkpoint": ("STRING", {
                    "default": "matrixgame2/base_distilled_model/base_distill.safetensors",
                    "tooltip": "Distilled generator weights. universal: "
                               "base_distilled_model/base_distill.safetensors, "
                               "gta_drive: gta_distilled_model/gta_keyboard2dim"
                               ".safetensors, templerun: templerun_distilled_"
                               "model/templerun_7dim_onlykey.safetensors",
                }),
                "vae_dir": ("STRING", {
                    "default": "matrixgame2",
                    "tooltip": "Directory holding Wan2.1_VAE.pth, "
                               "models_clip_open-clip-xlm-roberta-large-vit-"
                               "huge-14.pth and xlm-roberta-large/",
                }),
            },
            "optional": {
                "model_config_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing the DiT config.json. "
                               "Blank: configs/distilled_model/<mode> from the "
                               "source tree. Do not point this at the config"
                               ".json the HF repo ships next to base_distill "
                               "or gta_keyboard2dim: those omit "
                               "local_attn_size.",
                }),
                "config_yaml": ("STRING", {
                    "default": "",
                    "tooltip": "Inference yaml. Blank: "
                               "configs/inference_yaml/inference_<mode>.yaml "
                               "from the source tree.",
                }),
                "source_root": ("STRING", {
                    "default": "",
                    "tooltip": "Matrix-Game-2 source tree. Blank: the vendored "
                               "copy under mg2/Matrix-Game-2.",
                }),
                "compile_vae_decoder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "torch.compile the VAE decoder as upstream "
                               "does. Faster steady state, but the first job "
                               "pays a long autotune and it needs triton.",
                }),
            },
        }

    RETURN_TYPES = ("MG2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "Matrix-Game2"

    def load(self, mode, checkpoint, vae_dir, model_config_dir="",
             config_yaml="", source_root="", compile_vae_decoder=False):
        import torch

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        mg = _load_mg2(source_root)
        root = mg["root"]

        ckpt = _resolve_path(checkpoint, "checkpoint")
        vae = _resolve_path(vae_dir, "vae_dir")

        if config_yaml.strip():
            yaml_path = _resolve_path(config_yaml, "config_yaml")
        else:
            yaml_path = os.path.join(root, MODE_CONFIG_YAML[mode])
            if not os.path.isfile(yaml_path):
                raise FileNotFoundError(f"inference yaml missing: {yaml_path}")

        if model_config_dir.strip():
            cfg_dir = _resolve_path(model_config_dir, "model_config_dir")
        else:
            # Deliberately *not* the config.json next to the checkpoint. The
            # ones the HF repo ships for base_distilled_model and
            # gta_distilled_model are stale -- "_class_name": "WanModel" and no
            # local_attn_size at all, which lands on the CausalWanModel default
            # of -1 and trips CausalInferencePipeline's
            # `assert self.local_attn_size != -1`. The source tree's
            # configs/distilled_model/<mode> is the one that matches.
            cfg_dir = os.path.join(root, MODE_MODEL_CONFIG[mode])
        if not os.path.isfile(os.path.join(cfg_dir, "config.json")):
            raise FileNotFoundError(
                f"no config.json in model config dir: {cfg_dir}")

        key = (mode, ckpt, vae, cfg_dir, yaml_path, bool(compile_vae_decoder))
        with self._LOCK:
            bundle = self._CACHE.get(key)
            if bundle is None:
                while len(self._CACHE) >= self._CACHE_MAXSIZE:
                    _, evicted = self._CACHE.popitem()
                    evicted.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                bundle = self._build(mg, mode, ckpt, vae, cfg_dir, yaml_path,
                                     bool(compile_vae_decoder))
                self._CACHE[key] = bundle
        return (bundle,)

    @staticmethod
    def _build(mg, mode, ckpt, vae_dir, cfg_dir, yaml_path, compile_vae):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Matrix-Game 2.0 needs a CUDA device: the causal pipeline "
                "allocates its KV caches on the generator's device and the "
                "block VAE decoder runs in fp16."
            )
        device = torch.device("cuda")
        weight_dtype = torch.bfloat16

        with open(os.path.join(cfg_dir, "config.json"), "r", encoding="utf-8") as fh:
            model_config = json.load(fh)
        action_config = model_config.get("action_config") or {}
        keyboard_dim = int(action_config.get("keyboard_dim_in", 0))
        enable_mouse = bool(action_config.get("enable_mouse", False))
        if keyboard_dim != MODE_KEYBOARD_DIM[mode] or \
                enable_mouse != MODE_HAS_MOUSE[mode]:
            raise ValueError(
                f"model config {cfg_dir} declares keyboard_dim_in="
                f"{keyboard_dim}, enable_mouse={enable_mouse}, but mode "
                f"{mode!r} needs keyboard_dim_in="
                f"{MODE_KEYBOARD_DIM[mode]}, enable_mouse="
                f"{MODE_HAS_MOUSE[mode]}. Pick the mode that matches the "
                f"checkpoint, or point model_config_dir at the right config."
            )
        if int(model_config.get("local_attn_size", -1)) == -1:
            raise ValueError(
                f"{cfg_dir}/config.json has no usable local_attn_size. "
                f"CausalInferencePipeline asserts local_attn_size != -1, so "
                f"this is the foundation-model config (or the stale config.json "
                f"the HF repo ships next to base_distill/gta_keyboard2dim). Use "
                f"configs/distilled_model/{mode} from the source tree."
            )

        cfg_data = _read_yaml(yaml_path)
        yaml_mode = cfg_data.get("mode")
        if yaml_mode is not None and yaml_mode != mode:
            raise ValueError(
                f"{yaml_path} declares mode={yaml_mode!r} but the node was "
                f"asked for {mode!r}")
        model_kwargs = dict(cfg_data.get("model_kwargs") or {})
        # Upstream's yaml carries a *relative* model_config path and
        # WanDiffusionWrapper hands it straight to
        # CausalWanModel.from_config(). Two problems with that: it depends on
        # the server's cwd, and the non-dict argument goes through the
        # deprecated "pass a pretrained model name or path" branch of
        # ConfigMixin.from_config. Passing the parsed dict takes the supported
        # branch and cannot reach for the hub.
        model_kwargs["model_config"] = model_config
        cfg_data["model_kwargs"] = model_kwargs
        cfg = _Cfg(cfg_data)

        num_frame_per_block = int(getattr(cfg, "num_frame_per_block", 1))

        # generator: CausalWanModel built from the local config.json, weights
        # loaded from the local safetensors. Nothing touches the hub.
        generator = mg["WanDiffusionWrapper"](**model_kwargs, is_causal=True)

        state_dict = _load_state_dict(ckpt)
        _check_action_space(state_dict, ckpt, mode, keyboard_dim, enable_mouse)
        _load_generator_weights(generator, state_dict)

        # block VAE decoder: only decoder.* / conv2* live in the Wan VAE file
        vae_ckpt = os.path.join(vae_dir, "Wan2.1_VAE.pth")
        if not os.path.isfile(vae_ckpt):
            raise FileNotFoundError(f"missing {vae_ckpt}")
        vae_state_dict = torch.load(vae_ckpt, map_location="cpu")
        decoder_state_dict = {
            k: v for k, v in vae_state_dict.items()
            if "decoder." in k or "conv2" in k
        }
        vae_decoder = mg["VAEDecoderWrapper"]()
        vae_decoder.load_state_dict(decoder_state_dict)
        vae_decoder.to(device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        if compile_vae:
            vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        del vae_state_dict, decoder_state_dict

        pipeline = mg["CausalInferencePipeline"](
            cfg, generator=generator, vae_decoder=vae_decoder)
        pipeline = pipeline.to(device=device, dtype=weight_dtype)
        pipeline.vae_decoder.to(torch.float16)
        pipeline.eval()

        # encoder half: Wan2.1 VAE encoder + open-clip image encoder. Both are
        # plain torch.load from vae_dir; the tokenizer dir is local too.
        clip_ckpt = os.path.join(
            vae_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        # get_wanx_vae_wrapper joins ONE directory with all three of
        # Wan2.1_VAE.pth, the clip .pth and xlm-roberta-large/. The tokenizer
        # is four small json/model files, which the build scanner does not
        # treat as models and so never stages -- the staged volume has the two
        # .pth files and no tokenizer. It is 22MB, so it is vendored here, and
        # when the staged tree lacks it we compose a shadow directory of
        # symlinks so all three live under one path. The staged volume may be
        # read-only, hence a temp dir rather than writing next to the weights.
        tokenizer_dir = os.path.join(vae_dir, "xlm-roberta-large")
        if not os.path.exists(clip_ckpt):
            raise FileNotFoundError(f"missing {clip_ckpt}")
        if not os.path.isdir(tokenizer_dir):
            vendored = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "mg2", "tokenizer", "xlm-roberta-large")
            if not os.path.isdir(vendored):
                raise FileNotFoundError(
                    f"missing tokenizer dir: tried {tokenizer_dir} and {vendored}")
            import tempfile
            shadow = os.path.join(tempfile.gettempdir(), "mg2_vae_dir")
            os.makedirs(shadow, exist_ok=True)
            for name, target in (
                ("Wan2.1_VAE.pth", os.path.join(vae_dir, "Wan2.1_VAE.pth")),
                (os.path.basename(clip_ckpt), clip_ckpt),
                ("xlm-roberta-large", vendored),
            ):
                link = os.path.join(shadow, name)
                if not os.path.exists(link):
                    os.symlink(target, link)
            vae_dir = shadow
        vae = mg["get_wanx_vae_wrapper"](vae_dir, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, weight_dtype)

        return {
            "pipeline": pipeline,
            "vae": vae,
            "mode": mode,
            "keyboard_dim": keyboard_dim,
            "enable_mouse": enable_mouse,
            "num_frame_per_block": num_frame_per_block,
            "local_attn_size": int(model_config.get("local_attn_size", -1)),
            "device": device,
            "weight_dtype": weight_dtype,
            "checkpoint": ckpt,
            "model_config_dir": cfg_dir,
            "config_yaml": yaml_path,
            "source_root": mg["root"],
        }


def _load_state_dict(path: str) -> dict:
    import torch
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu")
    for k in ("state_dict", "module", "generator"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            break
    return obj


def _check_action_space(state_dict: dict, ckpt: str, mode: str,
                        keyboard_dim: int, enable_mouse: bool):
    """Confirm the weights themselves agree about the action space.

    ``action_model.keyboard_embed.0.weight`` is ``[hidden, keyboard_dim_in]``
    and ``action_model.mouse_mlp.0.weight`` is absent entirely when the
    checkpoint has no mouse branch, so the checkpoint answers the question
    directly. Checking here turns a mode/checkpoint mix-up into one clear
    message instead of a wall of load_state_dict shape errors.
    """
    kb_key = next((k for k in state_dict
                   if k.endswith("action_model.keyboard_embed.0.weight")), None)
    if kb_key is not None:
        found = int(state_dict[kb_key].shape[1])
        if found != keyboard_dim:
            raise ValueError(
                f"{os.path.basename(ckpt)} has a {found}-dim keyboard input "
                f"but mode {mode!r} expects {keyboard_dim}. universal=4, "
                f"gta_drive=2, templerun=7."
            )
    has_mouse = any(k.endswith("action_model.mouse_mlp.0.weight")
                    for k in state_dict)
    if has_mouse != enable_mouse:
        raise ValueError(
            f"{os.path.basename(ckpt)} "
            f"{'has' if has_mouse else 'has no'} mouse branch but mode "
            f"{mode!r} expects enable_mouse={enable_mouse}. Only templerun is "
            f"keyboard-only."
        )


def _load_generator_weights(generator, state_dict: dict):
    """Load into ``WanDiffusionWrapper`` or its inner ``CausalWanModel``.

    Upstream's checkpoints are saved from the wrapper, so keys are prefixed
    ``model.``; accepting the unprefixed form too costs one branch and makes a
    re-exported checkpoint work.
    """
    keys = list(state_dict.keys())
    prefixed = any(k.startswith("model.") for k in keys)
    target = generator if prefixed else generator.model
    try:
        target.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint does not match the model config. First keys: "
            f"{keys[:4]}. Underlying error: {e}"
        ) from e


class MatrixGame2ActionSequence:
    """Build the per-frame keyboard/mouse condition from JSON.

    The action space is per checkpoint, not global:

        universal   keyboard [T,4] = forward, back, left, right      + mouse
        gta_drive   keyboard [T,2] = forward, back                   + mouse
        templerun   keyboard [T,7] = nomove, jump, slide, turnleft,
                                     turnright, leftside, rightside  no mouse

    Keyboard entries are 0/1 flags. Mouse is ``[dpitch, dyaw]`` per frame in
    the units upstream's benchmark uses: +-0.1 for a camera nudge, 0 for none.

    ``num_output_frames`` counts *latent* frames and must be a multiple of the
    checkpoint's ``num_frame_per_block`` (3 for all released checkpoints). The
    action track is per *pixel* frame: ``(num_output_frames - 1) * 4 + 1``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(MODES), {
                    "default": "universal",
                    "tooltip": "Must match the loaded checkpoint; it sets the "
                               "keyboard width and whether mouse is used.",
                }),
                "num_output_frames": ("INT", {
                    "default": 150, "min": 3, "max": 900, "step": 3,
                    "tooltip": "Latent frames; multiple of 3. Pixel frames = "
                               "(n - 1) * 4 + 1, so 150 -> 597.",
                }),
                "keyboard": ("STRING", {
                    "multiline": True,
                    "default": "[1,0,0,0]",
                    "tooltip": "One row held for the clip, an explicit "
                               "per-frame list, or run-length segments like "
                               "[{\"repeat\":120,\"value\":[1,0,0,0]}]. Row "
                               "width must match the mode.",
                }),
                "mouse": ("STRING", {
                    "multiline": True,
                    "default": "[0,0]",
                    "tooltip": "[dpitch, dyaw] per frame, +-0.1 per step. "
                               "Ignored by templerun.",
                }),
            },
            "optional": {
                "pipeline": ("MG2_PIPELINE", {
                    "tooltip": "Optional: validates the action space against "
                               "the loaded checkpoint's config.json.",
                }),
            },
        }

    RETURN_TYPES = ("MG2_ACTIONS",)
    RETURN_NAMES = ("actions",)
    FUNCTION = "build"
    CATEGORY = "Matrix-Game2"

    def build(self, mode, num_output_frames, keyboard, mouse, pipeline=None):
        import torch

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        keyboard_dim = MODE_KEYBOARD_DIM[mode]
        enable_mouse = MODE_HAS_MOUSE[mode]
        block = 3
        if pipeline is not None:
            if pipeline["mode"] != mode:
                raise ValueError(
                    f"actions are for mode {mode!r} but the loaded pipeline is "
                    f"{pipeline['mode']!r}")
            keyboard_dim = pipeline["keyboard_dim"]
            enable_mouse = pipeline["enable_mouse"]
            block = pipeline["num_frame_per_block"]

        num_output_frames = int(num_output_frames)
        if num_output_frames < block or num_output_frames % block:
            raise ValueError(
                f"num_output_frames must be a positive multiple of the "
                f"checkpoint's num_frame_per_block ({block}), got "
                f"{num_output_frames}")

        frames = pixel_frames(num_output_frames)
        kb = _parse_action_json(keyboard, keyboard_dim, frames, "keyboard")
        kb_t = torch.tensor(kb, dtype=torch.float32).unsqueeze(0)

        ms_t = None
        if enable_mouse:
            ms = _parse_action_json(mouse, 2, frames, "mouse")
            ms_t = torch.tensor(ms, dtype=torch.float32).unsqueeze(0)

        return ({
            "mode": mode,
            "keyboard_dim": keyboard_dim,
            "enable_mouse": enable_mouse,
            "num_output_frames": num_output_frames,
            "num_frames": frames,
            "num_frame_per_block": block,
            "keyboard": kb_t,
            "mouse": ms_t,
        },)


class MatrixGame2ActionSampler:
    """Action-conditioned generation that returns frames as an IMAGE batch.

    Returning ``IMAGE`` rather than writing MP4s is the point: it is what makes
    a deployment job produce output assets through the normal contract. The
    output is ``[T, 352, 640, 3]``, float 0..1, on CPU. The resolution is not
    an input -- ``demo_utils.constant.ZERO_VAE_CACHE`` and the pipeline's
    ``frame_seq_length = 880`` are both built for a 44x80 latent.

    Each call is self-contained: ``CausalInferencePipeline.inference`` drops and
    reinitialises its KV caches at entry, and the VAE decoder's feature cache is
    local to the call. No state carries between jobs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("MG2_PIPELINE",),
                "start_image": ("IMAGE", {
                    "tooltip": "First frame. Centre-cropped to 640:352 and "
                               "resized to 352x640.",
                }),
                "actions": ("MG2_ACTIONS",),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
            },
            "optional": {
                "free_kv_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release the causal KV caches (~1.6GB) after "
                               "the job instead of holding them until the "
                               "next one reallocates.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "sample"
    CATEGORY = "Matrix-Game2"

    def sample(self, pipeline, start_image, actions, seed, free_kv_cache=True):
        import torch
        from einops import rearrange

        bundle = pipeline
        mode = bundle["mode"]
        device = bundle["device"]
        dtype = bundle["weight_dtype"]
        pipe = bundle["pipeline"]
        vae = bundle["vae"]

        if actions["mode"] != mode:
            raise ValueError(
                f"actions were built for mode {actions['mode']!r} but the "
                f"pipeline is {mode!r}")
        if actions["keyboard_dim"] != bundle["keyboard_dim"]:
            raise ValueError(
                f"actions have keyboard width {actions['keyboard_dim']} but "
                f"the checkpoint expects {bundle['keyboard_dim']}")
        if actions["enable_mouse"] != bundle["enable_mouse"]:
            raise ValueError(
                f"actions {'have' if actions['enable_mouse'] else 'lack'} a "
                f"mouse track but the checkpoint "
                f"{'needs' if bundle['enable_mouse'] else 'has no'} one")

        block = bundle["num_frame_per_block"]
        num_output_frames = int(actions["num_output_frames"])
        if num_output_frames % block:
            raise ValueError(
                f"num_output_frames {num_output_frames} is not a multiple of "
                f"num_frame_per_block {block}")
        num_frames = pixel_frames(num_output_frames)
        if actions["keyboard"].shape[1] != num_frames:
            raise ValueError(
                f"keyboard track has {actions['keyboard'].shape[1]} frames, "
                f"expected {num_frames}")

        # The pipeline draws its intermediate noise with torch.randn_like, so
        # reproducibility needs the global generator seeded as upstream's
        # set_seed does. The initial noise uses an explicit generator.
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        generator = torch.Generator(device=device).manual_seed(int(seed))

        image = _prep_start_image(start_image).to(device=device, dtype=dtype)

        with torch.no_grad():
            # Encode the first frame as latent 0 and zero-pad the rest, exactly
            # as inference.py does: the mask channels mark everything after
            # frame 0 invalid, so the padding's content does not matter, but its
            # length sets the latent length.
            padding_video = torch.zeros_like(image).repeat(
                1, 1, VAE_TIME_COMPRESSION * (num_output_frames - 1), 1, 1)
            img_cond = torch.concat([image, padding_video], dim=2)
            # tile_size/tile_stride are upstream's literals, in latent units.
            # The tile covers the whole 44x80 latent, so this is a single pass
            # with an all-ones blend mask.
            img_cond = vae.encode(
                img_cond, device=device,
                tiled=True, tile_size=[44, 80], tile_stride=[23, 38],
            ).to(device)
            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = vae.clip.encode_video(image)

            sampled_noise = torch.randn(
                [1, 16, num_output_frames, LATENT_H, LATENT_W],
                generator=generator, device=device, dtype=dtype,
            )

            conditional_dict = {
                "cond_concat": cond_concat.to(device=device, dtype=dtype),
                "visual_context": visual_context.to(device=device, dtype=dtype),
                "keyboard_cond": actions["keyboard"].to(device=device, dtype=dtype),
            }
            if bundle["enable_mouse"]:
                conditional_dict["mouse_cond"] = actions["mouse"].to(
                    device=device, dtype=dtype)

            try:
                videos = pipe.inference(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    return_latents=False,
                    mode=mode,
                    profile=False,
                )
                videos_tensor = torch.cat(videos, dim=1)  # [B, T, C, H, W]
                out = rearrange(videos_tensor, "B T C H W -> B T H W C")
                out = ((out.float() + 1.0) / 2.0).clamp(0, 1)[0].cpu()
            finally:
                del conditional_dict, cond_concat, img_cond, sampled_noise
                if free_kv_cache:
                    pipe.kv_cache1 = None
                    pipe.kv_cache_keyboard = None
                    pipe.kv_cache_mouse = None
                    pipe.crossattn_cache = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return (out,)


def _prep_start_image(image):
    """ComfyUI IMAGE [B,H,W,C] 0..1 -> [1, 3, 1, 352, 640] in -1..1.

    Mirrors upstream's ``_resizecrop`` + ``Resize((352, 640))`` +
    ``Normalize(0.5, 0.5)``.
    """
    import torch

    x = image[0:1].permute(0, 3, 1, 2).float()
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]
    if x.shape[1] != 3:
        raise ValueError(f"start_image must have 1, 3 or 4 channels, got "
                         f"{x.shape[1]}")

    h, w = x.shape[-2:]
    if h / w > PIXEL_H / PIXEL_W:
        new_w, new_h = w, int(round(w * PIXEL_H / PIXEL_W))
    else:
        new_h, new_w = h, int(round(h * PIXEL_W / PIXEL_H))
    new_h, new_w = max(1, min(h, new_h)), max(1, min(w, new_w))
    top, left = (h - new_h) // 2, (w - new_w) // 2
    x = x[:, :, top:top + new_h, left:left + new_w]

    x = torch.nn.functional.interpolate(
        x, size=(PIXEL_H, PIXEL_W), mode="bilinear",
        align_corners=False, antialias=True)
    x = x.clamp(0, 1) * 2.0 - 1.0
    return x.unsqueeze(2)


NODE_CLASS_MAPPINGS = {
    "MatrixGame2PipelineLoader": MatrixGame2PipelineLoader,
    "MatrixGame2ActionSequence": MatrixGame2ActionSequence,
    "MatrixGame2ActionSampler": MatrixGame2ActionSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatrixGame2PipelineLoader": "Matrix-Game 2.0 Pipeline Loader",
    "MatrixGame2ActionSequence": "Matrix-Game 2.0 Action Sequence",
    "MatrixGame2ActionSampler": "Matrix-Game 2.0 Action Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
