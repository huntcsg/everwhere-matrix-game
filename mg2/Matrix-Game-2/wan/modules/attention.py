# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name or "l20y" in device_name or "h800" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    # LOCAL PATCH (ComfyUI-Matrix-Game). flash-attn needs a CUDA compiler at
    # install time, so the deployment image does not have it and
    # matrixgame2_nodes._install_flash_attn_shim() registers an SDPA-backed
    # stand-in in sys.modules to satisfy action_module's module-scope
    # ``from flash_attn import flash_attn_func``. That stand-in has no varlen
    # kernel, so it must not be reported as FlashAttention-2: otherwise this
    # import succeeds, attention() routes into flash_attention(), and
    # flash_attn.flash_attn_varlen_func raises. Treat the stand-in as absent
    # so the SDPA paths below run instead.
    FLASH_ATTN_2_AVAILABLE = not (
        getattr(flash_attn, 'SDPA_SHIM', False)
        or getattr(flash_attn, '__version__', '') == '0.0.0')
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def _sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """LOCAL PATCH: scaled_dot_product_attention stand-in for flash_attention.

    Same [B, L, N, C] in/out contract as flash_attention. attention() has its
    own SDPA branch, but clip.py (and model.py) call flash_attention directly,
    so the fallback has to live here too.

    Only the argument combinations this tree actually uses are supported;
    anything SDPA cannot express faithfully raises rather than quietly
    returning a different result.
    """
    b, lq, nq = q.size(0), q.size(1), q.size(2)
    lk, nk = k.size(1), k.size(2)
    out_dtype = q.dtype

    if window_size != (-1, -1):
        raise NotImplementedError(
            'sliding-window attention has no scaled_dot_product_attention '
            f'equivalent (window_size={window_size})')
    if causal and lq != lk:
        # FlashAttention aligns a causal mask to the bottom right when the
        # query and key lengths differ; is_causal aligns it to the top left.
        # Silently picking the wrong one would corrupt the output, so refuse.
        raise NotImplementedError(
            f'causal attention with lq={lq} != lk={lk} needs a bottom-right '
            'aligned mask, which is_causal does not provide')
    if q_lens is not None:
        # flash_attention's own packed-return path cannot express non-uniform
        # q_lens either: it unflattens the packed output to (b, lq).
        warnings.warn(
            'Padding mask is disabled when using '
            'scaled_dot_product_attention. It can have a significant impact '
            'on performance.')

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if nq != nk:
        assert nq % nk == 0, f'Nq ({nq}) must be divisible by Nk ({nk})'
        k = k.repeat_interleave(nq // nk, dim=2)
        v = v.repeat_interleave(nq // nk, dim=2)

    attn_mask = None
    if k_lens is not None:
        # [B, 1, 1, Lk] boolean: drop each sequence's padding key columns.
        positions = torch.arange(lk, device=k.device)
        valid = positions[None, :] < k_lens.to(k.device).reshape(b, 1)
        attn_mask = valid[:, None, None, :]
        if causal:
            raise NotImplementedError(
                'k_lens together with causal would need the two masks merged')

    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal if attn_mask is None else False,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous().type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        # LOCAL PATCH: no flash-attn kernel in this image. Direct callers
        # (clip.py, model.py) never reach attention()'s own SDPA branch.
        return _sdpa_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
