"""Deployment-shaped nodes for Matrix-Game action-conditioned generation.

The upstream `GameVideoGenerator` node cannot be served from a Comfy API
deployment. Three reasons, all fatal:

  * it declares ``RETURN_TYPES = ()`` and writes MP4s to a local directory, so
    a job produces **no output assets** and the caller gets nothing back;
  * the action sequence is not an input at all -- it loads a hardcoded
    76-entry benchmark (``Bench_actions_76``) and loops over every one of
    them, so a single job renders 76 videos you did not ask for;
  * it mutates ``CUDA_VISIBLE_DEVICES`` at execution time, which is not safe
    inside a long-lived server process.

The *model* interface underneath is exactly what we want: per-frame
``keyboard_condition [B,T,6]`` and ``mouse_condition [B,T,2]``. That is a
genuine batched action sequence, so action-conditioned generation fits the
stateless ``/prompt`` -> assets contract without needing a live stream. This
module exposes that interface properly and returns an IMAGE batch, so the
standard SaveWEBM / SaveVideo / CreateVideo nodes emit real output assets.

Keyboard channels follow the upstream convention, one row per frame:
    [forward, back, left, right, jump, attack]
Mouse is [dx, dy] per frame in normalised units.
"""

from __future__ import annotations

import json

import numpy as np
import torch


def _parse_seq(raw: str, width: int, frames: int, name: str) -> torch.Tensor:
    """Parse an action sequence into a [1, frames, width] float tensor.

    Accepts either a full per-frame list (``[[1,0,0,0,0,0], ...]``) or a
    single row to hold for the whole clip (``[1,0,0,0,0,0]``). Shorter
    sequences are padded by repeating the last row, which is what you want
    when a caller describes an action that simply continues.
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
            raise ValueError(f"{name} must be a non-empty list")
        rows = parsed if isinstance(parsed[0], list) else [parsed]

    for r in rows:
        if len(r) != width:
            raise ValueError(f"{name} rows must have {width} values, got {len(r)}")

    if len(rows) < frames:
        rows = rows + [rows[-1]] * (frames - len(rows))
    rows = rows[:frames]

    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


class MatrixGameActionSequence:
    """Build keyboard+mouse conditions from JSON, so a caller can describe an
    action without needing to construct tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("INT", {"default": 65, "min": 5, "max": 1024}),
                "keyboard": ("STRING", {
                    "multiline": True,
                    "default": "[1,0,0,0,0,0]",
                    "tooltip": "[forward,back,left,right,jump,attack] per frame, "
                               "or one row to hold for the whole clip",
                }),
                "mouse": ("STRING", {
                    "multiline": True,
                    "default": "[0,0]",
                    "tooltip": "[dx,dy] per frame, or one row to hold",
                }),
            }
        }

    RETURN_TYPES = ("MG_ACTIONS",)
    RETURN_NAMES = ("actions",)
    FUNCTION = "build"
    CATEGORY = "Matrix-Game/everwhere"

    def build(self, frames, keyboard, mouse):
        kb = _parse_seq(keyboard, 6, frames, "keyboard")
        ms = _parse_seq(mouse, 2, frames, "mouse")
        return ({"keyboard": kb, "mouse": ms, "frames": frames},)


class MatrixGameActionSampler:
    """Action-conditioned clip generation that returns frames as IMAGE.

    Returning IMAGE rather than writing files is the whole point: it is what
    lets a deployment hand back output assets through the normal job contract.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("MG_PIPELINE",),
                "start_image": ("IMAGE",),
                "actions": ("MG_ACTIONS",),
                "width": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "sample"
    CATEGORY = "Matrix-Game/everwhere"

    def sample(self, pipeline, start_image, actions, width, height, steps,
               guidance_scale, seed):
        from einops import rearrange

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        kb = actions["keyboard"].to(dtype).to(device)
        ms = actions["mouse"].to(dtype).to(device)
        frames = int(actions["frames"])

        # ComfyUI IMAGE is [B,H,W,C] float 0..1; the pipeline wants a single
        # initial frame
        init = start_image[0:1]

        out = pipeline(
            height=height,
            width=width,
            video_length=frames,
            mouse_condition=ms,
            keyboard_condition=kb,
            initial_image=init,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            embedded_guidance_scale=None,
            data_type="video",
            vae_ver="884-16c-hy",
            enable_tiling=True,
            generator=torch.Generator(device=str(device)).manual_seed(seed),
            i2v_type="refiner",
            semantic_images=init,
        ).videos[0]

        # pipeline gives [C,T,H,W] in 0..1 -> ComfyUI wants [T,H,W,C]
        imgs = rearrange(out.permute(1, 0, 2, 3), "t c h w -> t h w c")
        return (imgs.float().clamp(0, 1).cpu(),)


class MatrixGameActionPreview:
    """Render the action track as a strip, so a profiling run can show what
    was asked for next to what came back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"actions": ("MG_ACTIONS",),
                             "height": ("INT", {"default": 64, "min": 16, "max": 256})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "Matrix-Game/everwhere"

    def render(self, actions, height):
        kb = actions["keyboard"][0].numpy()
        ms = actions["mouse"][0].numpy()
        t = kb.shape[0]
        rows = 8
        img = np.zeros((rows, t, 3), dtype=np.float32)
        for c in range(6):
            img[c, :, :] = kb[:, c][:, None]
        mx = np.abs(ms).max() or 1.0
        img[6, :, 0] = (ms[:, 0] / mx + 1) / 2
        img[7, :, 2] = (ms[:, 1] / mx + 1) / 2
        big = np.repeat(img, max(1, height // rows), axis=0)
        return (torch.from_numpy(big).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "MatrixGameActionSequence": MatrixGameActionSequence,
    "MatrixGameActionSampler": MatrixGameActionSampler,
    "MatrixGameActionPreview": MatrixGameActionPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatrixGameActionSequence": "Matrix-Game Action Sequence",
    "MatrixGameActionSampler": "Matrix-Game Action Sampler",
    "MatrixGameActionPreview": "Matrix-Game Action Preview",
}


class MatrixGamePipelineLoader:
    """Build the Matrix-Game pipeline once and cache it across jobs.

    Upstream rebuilds every model on each call because it was written as a
    one-shot CLI. In a deployment the worker is warm and serves many jobs, so
    a 17B DiT plus VAE plus text encoder must be constructed once and reused
    or every job pays the full load cost -- the difference between a ~10s
    clip and a ~2min one.
    """

    _CACHE: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_path": ("STRING", {"default": "models/matrixgame/dit"}),
                "vae_path": ("STRING", {"default": "models/matrixgame/vae"}),
                "textenc_path": ("STRING", {"default": "models/matrixgame"}),
                "teacache_thresh": ("FLOAT", {
                    "default": 0.075, "min": 0.0, "max": 1.0,
                    "tooltip": "0 disables TeaCache; higher skips more steps",
                }),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("MG_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "Matrix-Game/everwhere"

    def load(self, dit_path, vae_path, textenc_path, teacache_thresh, num_steps):
        key = (dit_path, vae_path, textenc_path)
        pipe = self._CACHE.get(key)
        if pipe is None:
            from matrixgame.sample.pipeline_matrixgame import MatrixGameVideoPipeline
            from matrixgame.model_variants import get_dit
            from matrixgame.vae_variants import get_vae
            from matrixgame.encoder_variants import get_text_enc
            from matrixgame.sample.flow_matching_scheduler_matrixgame import (
                FlowMatchDiscreteScheduler,
            )

            vae = get_vae("matrixgame", vae_path, torch.float16)
            vae.requires_grad_(False)
            vae.eval()
            vae.enable_tiling()

            # get_dit returns (model, class); it builds from_config, so the
            # weights are loaded separately by the caller upstream
            dit, _dit_cls = get_dit("matrixgame", dit_path, torch.bfloat16)
            dit.requires_grad_(False)
            dit.eval()

            text_enc = get_text_enc(
                "matrixgame", textenc_path, weight_dtype=torch.bfloat16,
                i2v_type="refiner",
            )

            pipe = MatrixGameVideoPipeline(
                vae=vae.vae,
                text_encoder=text_enc,
                transformer=dit,
                scheduler=FlowMatchDiscreteScheduler(shift=15.0, reverse=True,
                                                     solver="euler"),
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            self._CACHE[key] = pipe

        if teacache_thresh > 0:
            from teacache_forward import teacache_forward
            t = pipe.transformer.__class__
            t.enable_teacache = True
            t.cnt = 0
            t.num_steps = num_steps
            t.accumulated_rel_l1_distance = 0
            t.rel_l1_thresh = teacache_thresh
            t.previous_modulated_input = None
            t.previous_residual = None
            t.forward = teacache_forward
        return (pipe,)


NODE_CLASS_MAPPINGS["MatrixGamePipelineLoader"] = MatrixGamePipelineLoader
NODE_DISPLAY_NAME_MAPPINGS["MatrixGamePipelineLoader"] = "Matrix-Game Pipeline Loader"
