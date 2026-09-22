# Matrix-Game 2.0 node notes

Everything here was read out of the official source, not inferred. Upstream
paths below are relative to `Matrix-Game-2/` (vendored verbatim under
`mg2/Matrix-Game-2/`, see "Vendored source").

Nodes: `MatrixGame2PipelineLoader`, `MatrixGame2ActionSequence`,
`MatrixGame2ActionSampler` in `../matrixgame2_nodes.py`.

---

## 1. Action tensor contract, per checkpoint

The action space **differs per checkpoint**. The authority is
`action_config.keyboard_dim_in` / `action_config.enable_mouse` in the DiT
config, consumed by `wan/modules/action_module.py::ActionModule.__init__`. The
loader reads both and refuses a mode/checkpoint mismatch rather than assuming,
then cross-checks against the weights themselves — confirmed by reading the
safetensors headers of all three staged checkpoints:

```
base_distilled_model/base_distill.safetensors           1114 tensors
  action_model.keyboard_embed.0.weight [128, 4]   mouse_mlp.0.weight [1024, 1560]
gta_distilled_model/gta_keyboard2dim.safetensors        1114 tensors
  action_model.keyboard_embed.0.weight [128, 2]   mouse_mlp.0.weight [1024, 1560]
templerun_distilled_model/templerun_7dim_onlykey.safetensors  964 tensors
  action_model.keyboard_embed.0.weight [128, 7]   mouse_mlp.*  ABSENT
```

All three carry the action module in blocks 0..14 only, and all keys are
prefixed `model.`, i.e. saved from `WanDiffusionWrapper`, so
`pipeline.generator.load_state_dict(...)` is the correct target.
`mouse_mlp.0.weight` is `[1024, 2*4*3 + 1536]` = `[1024, 1560]`: the mouse
branch sees `mouse_dim_in * vae_time_compression_ratio * windows_size` = 24
action values concatenated with the 1536-dim image feature, which is why the
mouse track is per pixel frame rather than per latent frame.

| mode | checkpoint | keyboard | mouse | `local_attn_size` |
|---|---|---|---|---|
| `universal` | `base_distilled_model/base_distill.safetensors` | `[B, T, 4]` | `[B, T, 2]` | 6 |
| `gta_drive` | `gta_distilled_model/gta_keyboard2dim.safetensors` | `[B, T, 2]` | `[B, T, 2]` | 4 |
| `templerun` | `templerun_distilled_model/templerun_7dim_onlykey.safetensors` | `[B, T, 7]` | **none** | 6 |

`T` is the **pixel** frame count, not the latent count:

```
T = (num_output_frames - 1) * 4 + 1          # 150 latent frames -> 597
```

`num_output_frames` is the latent length, and must be a multiple of
`num_frame_per_block` (3 in all three inference yamls). Dtype/device at the
call site: same as the generator, i.e. `bfloat16` on `cuda`, batch 1.
`ActionModule.forward` asserts `(N_frames - 1) % 4 == 0` and
`(N_frames - 1)//4 + 1 == start_frame + num_frame_per_block`, which
`pipeline/causal_inference.py::cond_current` satisfies by slicing the full
track to `1 + 4*(current_start_frame + num_frame_per_block - 1)` each block.
So the node must hand over the **whole** track up front; it is sliced
internally, block by block.

### Channel semantics

One-hot / multi-hot 0-1 flags, from `utils/conditions.py` (`KEYBOARD_IDX`) and
`pipeline/causal_inference.py::get_current_action`:

* `universal` keyboard, index → key: `0 forward (W)`, `1 back (S)`,
  `2 left (A)`, `3 right (D)`. All-zero row = no movement. Multiple keys may
  be set at once (upstream's benchmark uses `forward_left` etc.).
* `gta_drive` keyboard: `0 forward (W)`, `1 back (S)`. All-zero = coast.
* `templerun` keyboard is one-hot over 7: `0 nomove (Q)`, `1 jump (W)`,
  `2 slide (S)`, `3 turnleft (Z)`, `4 turnright (C)`, `5 leftside (A)`,
  `6 rightside (D)`. Note the *upstream* label order here — `turnleft`/
  `turnright` sit at 3/4 and the side-steps at 5/6.
* mouse is `[d_pitch, d_yaw]` per frame. Upstream's `CAM_VALUE = 0.1` is the
  magnitude of a single camera nudge: `camera_up [+0.1, 0]`,
  `camera_down [-0.1, 0]`, `camera_left [0, -0.1]`, `camera_right [0, +0.1]`,
  diagonals combine both. `[0, 0]` = no camera motion. `gta_drive` only ever
  uses the yaw component in upstream's benchmark.
* For `templerun`, "no action" is **not** an all-zero row: `nomove` is a real
  channel at index 0, so standing still is `[1,0,0,0,0,0,0]`. An all-zero row
  is accepted by the model but was never trained as a state.
* Upstream holds each action for 12 pixel frames at a time
  (`utils/conditions.py::combine_data`, `selections = [12]`). Per-frame
  switching is legal but was not how the model was benchmarked; the node's
  run-length JSON form makes holding an action for N frames the easy thing.

### Node-level JSON forms

`MatrixGame2ActionSequence` accepts, for both `keyboard` and `mouse`:

* one row held for the whole clip — `[1,0,0,0]`
* explicit per-frame rows — `[[1,0,0,0],[0,0,1,0], ...]`
* run-length segments — `[{"repeat":120,"value":[1,0,0,0]},
  {"repeat":60,"value":[0,1,0,0]}]`

Short tracks are padded by repeating the last row; long tracks are truncated.
Row width is checked against the mode, and again against the checkpoint config
in the sampler. `mouse` is ignored for `templerun`.

---

## 2. Required weight layout

The loader takes `vae_dir` (a directory) and `checkpoint` (a file). Both accept
an absolute path, a path relative to ComfyUI's `models/` dir, or a path
relative to the ComfyUI root, so `matrixgame2/...` and
`models/matrixgame2/...` both work. Nothing is fetched from the hub; every
load is `torch.load` / `safetensors.load_file` / `AutoTokenizer` on a local
directory.

Stage the HuggingFace repo `Skywork/Matrix-Game-2.0` verbatim — its layout is
already exactly what the code wants:

```
models/matrixgame2/
  Wan2.1_VAE.pth                                            # encoder + block decoder
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth    # visual context encoder
  xlm-roberta-large/                                         # tokenizer dir (local only)
    sentencepiece.bpe.model
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
  base_distilled_model/
    base_distill.safetensors        + config.json            # universal
  gta_distilled_model/
    gta_keyboard2dim.safetensors    + config.json            # gta_drive
  templerun_distilled_model/
    templerun_7dim_onlykey.safetensors + config.json         # templerun
```

Who reads what:

* `vae_dir/Wan2.1_VAE.pth` — twice. `demo_utils/vae_block3.py::VAEDecoderWrapper`
  gets the `decoder.*` / `conv2*` subset (fp16, block-causal, rolling feature
  cache); `wan/vae/wanx_vae.py::get_wanx_vae_wrapper` loads the whole file for
  the encoder.
* `vae_dir/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` +
  `vae_dir/xlm-roberta-large/` — `wan/vae/wanx_vae_src/clip.py::CLIPModel`.
  `HuggingfaceTokenizer` calls `AutoTokenizer.from_pretrained(<dir>)`; with
  `tokenizer_config.json` + `tokenizer.json` present that resolves offline, no
  `config.json` needed.
* `checkpoint` — loaded into `WanDiffusionWrapper` (keys prefixed `model.`);
  the unprefixed form is accepted too and goes to the inner `CausalWanModel`.
* the DiT config — **read from the source tree**,
  `configs/distilled_model/<mode>/config.json`, not from the checkpoint's own
  directory. See the warning below. `model_config_dir` overrides it.

### Warning: the HF-shipped `config.json` is stale for two of the three

Read off the staged files:

| shipped config.json | `_class_name` | `local_attn_size` | usable? |
|---|---|---|---|
| `base_distilled_model/config.json` | `WanModel` | **absent** | no |
| `gta_distilled_model/config.json` | `WanModel` | **absent** | no |
| `templerun_distilled_model/config.json` | `CausalWanModel` | 6 | yes |

An absent `local_attn_size` lands on `CausalWanModel`'s default of `-1`, and
`CausalInferencePipeline.__init__` does `assert self.local_attn_size != -1`, so
loading `universal` or `gta_drive` from the HF config aborts. The source tree's
copies are the correct ones and disagree with each other, so the mode matters:

| mode | `local_attn_size` |
|---|---|
| `universal` | 6 |
| `gta_drive` | **4** |
| `templerun` | 6 |

The loader raises a named error if the resolved config has no usable
`local_attn_size`, rather than letting the assert fire deep in the pipeline.
Staging the HF `config.json` files is harmless — nothing reads them — but do
not point `model_config_dir` at them.

`base_model/` (the non-distilled foundation model) and `model_index.json` from
the HF repo are **not** used: there is no inference yaml for the foundation
model, its `config.json` has `local_attn_size: -1`, and
`CausalInferencePipeline.__init__` asserts `local_attn_size != -1`. Staging it
is wasted disk.

Output resolution is fixed at **352x640** (44x80 latent). Not a knob:
`demo_utils/constant.py::ZERO_VAE_CACHE` is built for a 44x80 latent and
`CausalInferencePipeline.frame_seq_length = 880 = 44*80/4`. The node
centre-crops and resizes `start_image` to match, as `inference.py` does.

---

## 3. Config file per checkpoint

`configs/inference_yaml/` supplies the sampler schedule; the loader reads it
from the source tree (override with `config_yaml`).

| mode | inference yaml | denoising steps | `num_frame_per_block` | DiT config |
|---|---|---|---|---|
| `universal` | `inference_universal.yaml` | `[1000, 666, 333]` | 3 | `configs/distilled_model/universal/config.json` |
| `gta_drive` | `inference_gta_drive.yaml` | `[1000, 666, 333]` | 3 | `configs/distilled_model/gta_drive/config.json` |
| `templerun` | `inference_templerun.yaml` | `[1000, 750, 500, 250]` | 3 | `configs/distilled_model/templerun/config.json` |

All three yamls also set `warp_denoising_step: true`, `context_noise: 0` and
`model_kwargs.timestep_shift: 5.0`.

The yaml's `model_kwargs.model_config` is a **relative** path
(`configs/distilled_model/<mode>`). The loader overwrites it with the absolute
config directory it validated — relying on it would mean depending on the
server's cwd.

`configs/foundation_model/config.json` corresponds to `base_model/`, not to any
distilled checkpoint: it has `local_attn_size: -1` and the action module in all
30 blocks, so it cannot be loaded by `CausalInferencePipeline` either.

---

## 4. Compiler-dependent dependencies, guarded

Checked every import in the tree (`grep -rn "flash_attn\|apex\|tensorrt\|pycuda\|onnx"`).

* **`flash_attn` — hard import, guarded.** `wan/modules/action_module.py:3`
  does `from flash_attn import flash_attn_func` at module scope, and the
  KV-cache path at lines 313/321/440/449/514/523 calls it for real. flash-attn
  needs a CUDA compiler at install time, so `matrixgame2_nodes.py`
  `_install_flash_attn_shim()` registers an SDPA-backed `sys.modules`
  ["flash_attn"] with a matching `[B, L, H, D]` signature before the vendored
  tree is imported. Same approach as
  `matrixgame/model_variants/matrixgame_dit_src/_attn_compat.py` in the 1.0
  tree, but injected as a module so the vendored source stays byte-identical
  to upstream. If a real wheel is present it wins (`find_spec` check).
* **`wan/modules/attention.py` and `wan/vae/wanx_vae_src/attention.py` needed
  patching, and an earlier revision of this file was wrong to say they did
  not.** Both open with

  ```python
  try:
      import flash_attn
      FLASH_ATTN_2_AVAILABLE = True
  except ModuleNotFoundError:
      FLASH_ATTN_2_AVAILABLE = False
  ```

  The `try/except` guards an *absent* flash-attn. It does not guard a shim: the
  import succeeds, `FLASH_ATTN_2_AVAILABLE` becomes True, and both the
  `attention()` router and `flash_attention()` itself go to
  `flash_attn.flash_attn_varlen_func`, which the shim does not implement. The
  shim made the two files' own SDPA fallbacks unreachable.

  Both now test for the shim (`SDPA_SHIM`, or `__version__ == "0.0.0"`) and
  report `FLASH_ATTN_2_AVAILABLE = False`, so `attention()` takes upstream's
  own SDPA branch.

  **Flipping the flag is not sufficient on its own**, which is the part that is
  easy to miss. `flash_attention()` is not only reached through `attention()`
  — `wan/vae/wanx_vae_src/clip.py:91,203` and `wan/modules/model.py:150,255`
  call it *directly*, and with the flag False it would hit its own
  `assert FLASH_ATTN_2_AVAILABLE` on line ~112. So `flash_attention()` also
  got an early `_sdpa_attention()` branch, before the varlen packing, where
  q/k/v are still `[B, L, N, C]`.

  `_sdpa_attention` honours `softmax_scale`, `q_scale`, `causal`, `k_lens`
  (as a key-padding mask) and `Nq % Nk == 0` head broadcasting, preserves the
  input dtype on the way out, and **raises** rather than approximating for
  `window_size != (-1, -1)` and for `causal` with `lq != lk` (FlashAttention
  aligns a causal mask bottom-right when the lengths differ, `is_causal`
  aligns top-left). No call site in this tree uses either, and wrong attention
  is wrong video, which is worse than a crash.

  Which one actually bites first: **`wan/vae/wanx_vae_src/attention.py`**.
  `MatrixGame2ActionSampler.sample` calls `vae.clip.encode_video(image)`
  before `pipe.inference`, and that is `CLIPModel` →
  `VisionTransformer` → `SelfAttention` → `flash_attention(..., version=2)`
  in the VAE's copy. `wan/modules/attention.py` is reached later, from
  `causal_model.py:187`, and only through `attention()`.

  Verified numerically against an explicit `softmax(QK^T·scale)V` reference
  (not against SDPA, which would pass by construction) for both files and for
  the shim's own `flash_attn_func`: `mg2/test_sdpa_fallback.py`, CPU torch is
  enough. bf16 agreement ~4e-3, fp32 ~1e-7.
* `flash_attn_interface` (FA3) is `try/except` everywhere. Left absent.
* **apex** — not imported anywhere in the 2.0 tree, only mentioned in the
  README install steps. Nothing to guard.
* **tensorrt / pycuda / torch2trt** — `demo_utils/vae.py:3` (`import tensorrt`)
  and `demo_utils/vae_torch2trt.py` (`pycuda`, `tensorrt`, `onnxruntime`) are
  hard imports, but they belong to the TensorRT VAE variant that
  `inference.py` does not use. The node imports `demo_utils.vae_block3` and
  `demo_utils.constant` by module path and never touches `demo_utils/__init__`
  (there isn't one) or `demo_utils/vae.py`, so those imports never execute.
  `nvidia-tensorrt` / `pycuda` / `onnx*` in `requirements.txt` are **not
  needed** and should not be installed — `nvidia-tensorrt` in particular
  builds from source.
* `opencv-python` — `utils/visualize.py` imports `cv2` for the MP4 overlay
  helpers, and `pipeline/causal_inference.py` imports it at module scope.
  Those helpers are exactly what these nodes replace, so
  `_stub_visualize_if_no_cv2()` installs a raising stub for `utils.visualize`
  when `cv2` is missing rather than letting the pipeline import fail. (ComfyUI
  normally ships `opencv-python-headless`, which satisfies it for real.)
* Other `requirements.txt` entries that are **not** needed on this path and
  can be skipped: `dashscope`, `wandb`, `flask`, `flask-socketio`,
  `pycocotools`, `lmdb`, `dominate`, `nvidia-pyindex`, `torchao`,
  `git+https://github.com/openai/CLIP.git`, `open_clip_torch` (the CLIP
  implementation is vendored in `wan/vae/wanx_vae_src/clip.py`).
* Runtime deps that **are** needed beyond ComfyUI's own: `diffusers` (for
  `ConfigMixin` / `ModelMixin` on `CausalWanModel` and `CLIPModel`),
  `transformers` (`AutoTokenizer`), `einops`, `safetensors`, `ftfy`, `regex`,
  `PyYAML`, `torchvision`. `omegaconf` is **not** required — the node reads the
  yaml with PyYAML into a plain namespace. No numpy pin.
* **The shim needs `__spec__`.** transformers probes with
  `importlib.util.find_spec("flash_attn")`, and `find_spec` raises
  `ValueError: flash_attn.__spec__ is None` for a module that sits in
  `sys.modules` without one. The shim therefore carries a real `ModuleSpec`.

  What actually keeps the FlashAttention-2 gates closed is **not**
  `__version__`. transformers 4.47.1's `_is_package_available` calls
  `importlib.metadata.version("flash_attn")`, which raises
  `PackageNotFoundError` because no distribution is installed, and the helper
  then reports the package absent — so `is_flash_attn_2_available()`,
  `is_flash_attn_greater_or_equal_2_10()` and `is_flash_attn_greater_or_equal()`
  all answer False. Checked by running transformers' own logic against the
  installed shim. `__version__ = "0.0.0"` is therefore only a shim *marker*,
  which is why the vendored patch keys off `SDPA_SHIM` first and treats
  `"0.0.0"` as a secondary tell.

  diffusers 0.32.2 (the pinned version) contains **zero** `flash_attn`
  references — `grep -rn flash_attn` over the wheel is empty — so the earlier
  note blaming diffusers for the `__spec__` requirement was wrong about which
  library probes. peft 0.14.0 likewise has none.

* **ComfyUI core has the same pattern and is safe only by ordering.**
  `comfy/ldm/modules/attention.py:47` does `from flash_attn import
  flash_attn_func` at module scope and sets `FLASH_ATTENTION_IS_AVAILABLE`.
  That runs at ComfyUI startup, long before `_load_mg2()` installs the shim on
  first node execution, so ComfyUI records flash-attn as absent and keeps it
  that way; it is additionally only used behind `--use-flash-attention`.
  **Do not move the shim installation to import time** — it would flip
  ComfyUI's own flag and route unrelated models into an SDPA stand-in.

* `flash_attn_qkvpacked_func` used to be aliased to the shim's
  `flash_attn_func`. The real one takes a single packed `[B, L, 3, H, D]`
  tensor, so the alias was a wrong-signature trap; it now raises. Nothing in
  the tree calls it (`grep -rn qkvpacked` is empty).

* `wan/utils/prompt_extend.py:17` also imports `flash_attn_varlen_func`, in a
  `try/except`. Nothing imports that module (only `wan/__init__.py` would, and
  it never runs), so the name binds to the raising stub and is never called.
* **`wan/modules/t5.py` demands a GPU at import time.** Line 478 calls
  `torch.cuda.current_device()` in a *class body*, so merely executing
  `wan/modules/__init__.py` (which does `from .t5 import ...`) raises
  `AssertionError: Torch not compiled with CUDA enabled` on a CPU host. This
  path does not use T5 at all, so `wan.modules` is claimed as a namespace
  package and that `__init__` never runs.
* `wan/modules/posemb_layers.py::get_meshgrid_nd` hardcodes
  `device=torch.cuda.current_device()`, so `ActionModule.__init__` — and hence
  the DiT — cannot be constructed on CPU at all. The loader requires CUDA and
  says so; there is no CPU fallback to offer.
* `torch.nn.attention.flex_attention` (torch >= 2.5) is imported at module
  scope by `wan/modules/causal_model.py` and `wan/modules/action_module.py`,
  and `action_module` wraps it in `torch.compile(..., mode=
  "max-autotune-no-cudagraphs")` at import time. The wrap is lazy and the
  compiled function is only called on the *non*-KV-cache branch, which
  `CausalInferencePipeline` never takes, so triton is not needed at inference.
  `compile_vae_decoder` (default off) is the only thing that would pull triton
  in.

---

## 5. Import plumbing, and why it is not `sys.path`

The vendored tree uses absolute imports for its own siblings — `from
utils.wan_wrapper import ...`, `from wan.modules.model import ...`, `from
demo_utils.constant import ZERO_VAE_CACHE`. Two of those top-level names are
already taken in a ComfyUI process: **ComfyUI ships its own `utils/` package**
(`utils/extra_config.py`, `utils/json_util.py`, ...) and imports it at startup,
so naively putting the source root on `sys.path` gives
`ModuleNotFoundError: No module named 'utils.wan_wrapper'` — whichever `utils`
is found first wins and it is not ours.

`_claim_package()` therefore appends the vendored directory to the existing
package's `__path__` (and synthesises a namespace package when the name is
free), touching neither `sys.path` nor ComfyUI's `utils`. The submodule names
do not overlap, so both halves keep resolving. Verified:
`utils.json_util` → ComfyUI, `utils.wan_wrapper` → vendored.

A side benefit: `wan` is claimed as a synthesised namespace package, so
`wan/__init__.py` never runs and `WanI2V` / `WanT2V` / `T5EncoderModel` /
`prompt_extend` (which imports `dashscope`) are never imported.

---

## 6. Vendored source

`mg2/Matrix-Game-2/` is `wan/`, `utils/`, `pipeline/`, `demo_utils/` and
`configs/` copied from the official repo (MIT) so it stays diffable against
upstream. **Two files carry a local patch**, each marked `LOCAL PATCH` at
every hunk:

```
wan/modules/attention.py              shim detection + flash_attention SDPA fallback
wan/vae/wanx_vae_src/attention.py     same
diff -r <official>/Matrix-Game-2 mg2/Matrix-Game-2   # only those two files
```

Both patches are additive — a new `_sdpa_attention` helper, one changed
assignment in the `import flash_attn` guard, and one early-return branch. No
upstream line was deleted and `attention()`'s own SDPA branch is untouched, so
a future upstream bump can drop the patch the moment a real flash-attn wheel
is in the image (the guard then reports True again and nothing else changes).
See §4 for why it is needed.

It is vendored rather than referenced because the deployment image stages the
ComfyUI tree, not the sibling Matrix-Game clone. The loader still prefers an
explicit `source_root` input or `$MATRIXGAME2_SRC` if either is set, then the
vendored copy, then a few sibling locations.

Not vendored (unused by this path): `inference.py`,
`inference_streaming.py`, `assets/`, `demo_images/`, `setup.py`.

---

## 7. Statelessness of the request/response path

`CausalInferencePipeline.inference` (`pipeline/causal_inference.py:216`) opens
with

```python
self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache = None
```

and then reallocates all four caches from scratch, so **no KV state carries
between calls** and the `/prompt` → assets contract is safe. The VAE decoder's
`feat_cache` is a local `copy.deepcopy(ZERO_VAE_CACHE)` per call, likewise.
The sampler clears the four cache attributes again after the job (~1.6GB of
`local_attn_size`-sized buffers) so a warm worker does not sit on them.

What genuinely does **not** fit request/response:
`CausalInferenceStreamingPipeline` (same file) calls `get_current_action()` →
`input()` on stdin between every 3-frame block, and `input("Continue?")` after
each one. That is interactive by construction — in a server it would block the
worker on a read that never returns. It is not wrapped, and real interactive
streaming would need a session-scoped pipeline plus a transport that can feed
actions mid-generation, i.e. not a Comfy `/prompt`. The batch pipeline used
here takes the entire action track up front instead, which is the same model
and the same conditioning, just decided before the job starts rather than
during it.

---

## 8. What was actually verified, and what was not

Verified on this box (no GPU, CPU-only torch 2.14 + diffusers 0.40 +
transformers 5.17 installed for the purpose):

* `mg2/test_sdpa_fallback.py` passes: 24 numeric and behavioural checks over
  both patched `attention.py` files and the installed shim, against an
  explicit `softmax(QK^T·scale)V` reference rather than against SDPA. Covers
  the exact call shapes this tree produces — clip self-attention (`lq == lk`),
  clip attention-pool (`lq = 1`), the causal-model KV-cache window
  (`lq != lk`), `softmax_scale`, `q_scale`, `k_lens`, GQA head broadcasting,
  dtype round-trip — and asserts the two unsupported cases raise;

* the required import check passes with **no torch at all installed** — every
  heavy import in `matrixgame2_nodes.py` is deferred to execution, so
  `INPUT_TYPES()` is honest and ComfyUI still starts if the environment is
  incomplete;
* the whole vendored import chain resolves under the `_claim_package` scheme
  with ComfyUI's own `utils` imported first: `wan.modules.action_module`,
  `pipeline.causal_inference` (both pipeline classes), `demo_utils.vae_block3`,
  `utils.wan_wrapper` — and `utils.json_util` still resolves to ComfyUI's copy
  afterwards;
* `ActionModule` built from each mode's real `action_config` accepts the exact
  tensors `MatrixGame2ActionSequence` produces, sliced by upstream's own
  `cond_current`, across three blocks with KV-cache rolling, output
  `[1, 2640, 1536]` and finite for all three modes. The per-block slices come
  out 9 / 21 / 33 frames for a 9-latent-frame clip, and `local_end_index` caps
  at `local_attn_size` (4 for gta_drive, 6 for the others) as expected;
* the safetensors headers of all three staged checkpoints agree with the
  action-space table above;
* **both `load_state_dict` calls in the loader match the real staged weights,
  key for key and shape for shape.** Building
  `WanDiffusionWrapper(model_config=<parsed dict>, timestep_shift=5.0,
  is_causal=True)` on the `meta` device and diffing its `state_dict()` against
  each checkpoint's safetensors header: 1114/1114 for `base_distill`, 1114/1114
  for `gta_keyboard2dim`, 964/964 for `templerun_7dim_onlykey` — zero missing,
  zero extra, zero shape mismatches, so `strict=True` succeeds. Same check for
  `VAEDecoderWrapper` against the `decoder.*` / `conv2*` subset of
  `Wan2.1_VAE.pth`: 108/108 of the file's 194 tensors, exact;
* passing the parsed config **dict** to `CausalWanModel.from_config` yields a
  byte-identical init dict to the deprecated path form
  (`load_config(pretrained_model_name_or_path=<dir>)`) for all three modes, so
  the loader takes the supported, provably network-free branch. `from_config`
  drops `inject_sample_info` as unexpected in both forms — upstream behaviour,
  not a symptom;
* `_prep_start_image` returns `[1, 3, 1, 352, 640]` in `[-1, 1]` for 1-, 3- and
  4-channel inputs at several aspect ratios;
* the output conversion yields `[T, 352, 640, 3]`, float32, CPU, within
  `[0, 1]`, with `-1 -> 0`, `0 -> 0.5`, `1 -> 1`;
* the pack's `__init__.py` registers all three new nodes alongside the existing
  1.0 ones.

**Not** verified, for lack of a GPU: an end-to-end generation. The DiT weights
load, the VAE halves load and the pipeline is constructed following
`inference.py` step for step, but nothing here has run the 30-block model, the
Wan VAE encoder, the CLIP image encoder or the block decoder on real weights.
The first GPU run is where a dtype or device mismatch in the loader would show
up.

---

One cost worth knowing: `inference.py`'s conditioning trick, reproduced here,
VAE-encodes a full-length video (`1 + 4*(num_output_frames-1)` frames, 597 at
the default) of which every frame after the first is zero, purely to get the
latent length right — the mask channels mark them invalid. At 597 frames that
is a real chunk of the wall clock, and `wan/vae/wanx_vae_src/vae.py::encode`
routes the pixel tensor through CPU on the way in. Faithful to upstream, but it
is the obvious place to look if a job is slower than the DiT time suggests.
