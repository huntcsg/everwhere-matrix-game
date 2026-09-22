"""flash-attn compatibility shim.

`flash_attn` needs a CUDA compiler at install time, which makes it a poor
dependency for a reproducible image build. PyTorch's
`scaled_dot_product_attention` reaches the same FlashAttention kernels on
Ampere and newer anyway, so when the package is missing we substitute an
SDPA implementation with a matching signature rather than fail to import.

flash_attn_func takes and returns [B, L, H, D]; SDPA wants [B, H, L, D].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # real kernel when the wheel is present
    from flash_attn import flash_attn_func as _real_flash_attn_func
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _real_flash_attn_func = None

HAVE_FLASH_ATTN = _real_flash_attn_func is not None


def _sdpa_flash_attn_func(
    q,
    k,
    v,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_ignored,
):
    # [B, L, H, D] -> [B, H, L, D]
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(
        qt,
        kt,
        vt,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


flash_attn_func = _real_flash_attn_func or _sdpa_flash_attn_func
