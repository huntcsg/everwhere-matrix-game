"""Check the local SDPA fallback in both vendored attention.py files.

_sdpa_attention stands in for flash_attention, which for the call sites in this
tree is plain dense attention. The reference here is an explicit softmax(QK^T/
sqrt(d))V written out per batch/head, so a transpose or scale slip shows up as
a numeric difference rather than passing by construction.

Run: python mg2/test_sdpa_fallback.py    (CPU torch is enough)
"""
import importlib.machinery
import importlib.util
import math
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Matrix-Game-2")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_shim():
    """The same stand-in matrixgame2_nodes._install_flash_attn_shim registers."""
    mod = types.ModuleType("flash_attn")

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                        causal=False, **_ignored):
        qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, dropout_p=0.0, is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2)

    def flash_attn_varlen_func(*_a, **_k):
        raise RuntimeError("varlen tripwire reached")

    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    mod.__version__ = "0.0.0"
    mod.SDPA_SHIM = True
    sys.modules["flash_attn"] = mod


def reference(q, k, v, softmax_scale=None, q_scale=None, causal=False,
              k_lens=None, dtype=torch.float64):
    """softmax(QK^T * scale)V, written out. q/k/v are [B, L, N, C]."""
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if q_scale is not None:
        q = q * q_scale
    b, lq, nq, c = q.shape
    lk, nk = k.shape[1], k.shape[2]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(c)
    out = torch.zeros(b, lq, nq, v.shape[-1], dtype=dtype)
    for bi in range(b):
        for hi in range(nq):
            hk = hi // (nq // nk)
            logits = (q[bi, :, hi] @ k[bi, :, hk].T) * scale
            if causal:
                assert lq == lk
                mask = torch.triu(torch.ones(lq, lk, dtype=torch.bool), 1)
                logits = logits.masked_fill(mask, float("-inf"))
            if k_lens is not None:
                invalid = torch.arange(lk) >= int(k_lens[bi])
                logits = logits.masked_fill(invalid[None, :], float("-inf"))
            out[bi, :, hi] = logits.softmax(-1) @ v[bi, :, hk]
    return out


def check(name, got, want, tol):
    err = (got.to(torch.float64) - want).abs().max().item()
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<52} max|err|={err:.3e}")
    return ok


def check_real_shim():
    """The shim matrixgame2_nodes actually installs, not the copy above.

    action_module.py calls flash_attn_func on the KV-cache path for real, and
    every one of its call sites is non-causal with q/k head counts equal, so
    the only thing that can be wrong is the [B,L,H,D] <-> [B,H,L,D] transpose.
    """
    print("\nmatrixgame2_nodes._install_flash_attn_shim()")
    fails = []
    del sys.modules["flash_attn"]
    sys.path.insert(0, os.path.dirname(HERE))
    import matrixgame2_nodes

    matrixgame2_nodes._install_flash_attn_shim()
    shim = sys.modules["flash_attn"]

    if not getattr(shim, "SDPA_SHIM", False):
        fails.append("shim: SDPA_SHIM flag missing (the vendored patch keys off it)")
        print("  FAIL SDPA_SHIM flag missing")
    else:
        print("  ok   SDPA_SHIM flag present")
    if importlib.util.find_spec("flash_attn") is None:
        fails.append("shim: find_spec() returns None")
    else:
        print("  ok   find_spec('flash_attn') resolves (transformers probes it)")

    torch.manual_seed(1)
    # action_module mouse branch: q is one block of frames, k/v the cache
    # window. [B*S, T, H, D].
    q = torch.randn(6, 3, 4, 8)
    k = torch.randn(6, 6, 4, 8)
    v = torch.randn(6, 6, 4, 8)
    got = shim.flash_attn_func(q, k, v)
    if tuple(got.shape) != (6, 3, 4, 8):
        fails.append(f"shim: shape {tuple(got.shape)}")
    ok = check("flash_attn_func lq=3 lk=6 (KV cache window)", got,
               reference(q, k, v), 4e-3)
    if not ok:
        fails.append("shim: flash_attn_func")

    q2 = torch.randn(2, 9, 4, 8)
    got = shim.flash_attn_func(q2, q2, q2, causal=True)
    ok = check("flash_attn_func causal=True", got,
               reference(q2, q2, q2, causal=True), 4e-3)
    if not ok:
        fails.append("shim: flash_attn_func causal")

    got = shim.flash_attn_func(q2, q2, q2, softmax_scale=0.31)
    ok = check("flash_attn_func softmax_scale honoured", got,
               reference(q2, q2, q2, softmax_scale=0.31), 4e-3)
    if not ok:
        fails.append("shim: flash_attn_func softmax_scale")

    # The varlen tripwire must still be a tripwire: nothing should reach it
    # now, and if something does it must crash rather than guess.
    try:
        shim.flash_attn_varlen_func(q, k, v)
        print("  FAIL flash_attn_varlen_func returned instead of raising")
        fails.append("shim: varlen tripwire disarmed")
    except RuntimeError:
        print("  ok   flash_attn_varlen_func still raises (tripwire intact)")
    return fails


def main():
    install_shim()
    torch.manual_seed(0)
    fails = []

    for label, path in (
        ("wan/modules", os.path.join(SRC, "wan", "modules", "attention.py")),
        ("wan/vae/wanx_vae_src",
         os.path.join(SRC, "wan", "vae", "wanx_vae_src", "attention.py")),
    ):
        mod = load(path, f"attn_{label.replace('/', '_')}")
        print(f"\n{label}/attention.py")

        # The whole point: the shim must not be reported as FlashAttention-2.
        if mod.FLASH_ATTN_2_AVAILABLE is not False:
            fails.append(f"{label}: FLASH_ATTN_2_AVAILABLE is not False")
            print("  FAIL FLASH_ATTN_2_AVAILABLE is not False")
        else:
            print("  ok   FLASH_ATTN_2_AVAILABLE is False with the shim loaded")

        # bf16 in, bf16 out, ~3 decimal digits of mantissa.
        tol = 3e-2
        f32tol = 4e-3

        # 1. clip.py SelfAttention: [B,L,N,C], lq == lk, non-causal, no lens.
        q = torch.randn(2, 17, 4, 8, dtype=torch.bfloat16)
        k = torch.randn(2, 17, 4, 8, dtype=torch.bfloat16)
        v = torch.randn(2, 17, 4, 8, dtype=torch.bfloat16)
        got = mod.flash_attention(q, k, v, version=2)
        if got.dtype is not torch.bfloat16:
            fails.append(f"{label}: out_dtype not preserved ({got.dtype})")
        if tuple(got.shape) != (2, 17, 4, 8):
            fails.append(f"{label}: shape {tuple(got.shape)}")
        fails += [] if check("clip SelfAttention (lq==lk, non-causal)", got,
                             reference(q, k, v), tol) else [f"{label}: self-attn"]

        # 2. clip.py AttentionPool: lq=1, lk=s, non-causal.
        q1 = torch.randn(2, 1, 4, 8, dtype=torch.bfloat16)
        got = mod.flash_attention(q1, k, v, version=2)
        fails += [] if check("clip AttentionPool (lq=1, lk=17)", got,
                             reference(q1, k, v), tol) else [f"{label}: pool"]

        # 3. causal_model.py attention(): defaults, lq==lk.
        got = mod.attention(q, k, v)
        fails += [] if check("attention() default (upstream SDPA branch)", got,
                             reference(q, k, v), tol) else [f"{label}: attention()"]

        # 4. causal_model.py KV-cache shape: lq != lk, non-causal.
        kk = torch.randn(2, 40, 4, 8, dtype=torch.bfloat16)
        vv = torch.randn(2, 40, 4, 8, dtype=torch.bfloat16)
        got = mod.attention(q, kk, vv)
        fails += [] if check("attention() lq=17 lk=40 (KV cache window)", got,
                             reference(q, kk, vv), tol) else [f"{label}: attn kv"]

        # 5. causal, lq == lk.
        qf = torch.randn(2, 17, 4, 8)
        kf = torch.randn(2, 17, 4, 8)
        vf = torch.randn(2, 17, 4, 8)
        got = mod.flash_attention(qf, kf, vf, causal=True, dtype=torch.float32,
                                  version=2)
        fails += [] if check("causal=True, lq==lk", got,
                             reference(qf, kf, vf, causal=True),
                             f32tol) else [f"{label}: causal"]

        # 6. softmax_scale and q_scale are honoured, not dropped.
        got = mod.flash_attention(qf, kf, vf, softmax_scale=0.37,
                                  dtype=torch.float32, version=2)
        fails += [] if check("softmax_scale=0.37 honoured", got,
                             reference(qf, kf, vf, softmax_scale=0.37),
                             f32tol) else [f"{label}: softmax_scale"]
        got = mod.flash_attention(qf, kf, vf, q_scale=2.5,
                                  dtype=torch.float32, version=2)
        fails += [] if check("q_scale=2.5 honoured", got,
                             reference(qf, kf, vf, q_scale=2.5),
                             f32tol) else [f"{label}: q_scale"]

        # 7. k_lens padding mask.
        k_lens = torch.tensor([17, 9], dtype=torch.int32)
        got = mod.flash_attention(qf, kf, vf, k_lens=k_lens,
                                  dtype=torch.float32, version=2)
        fails += [] if check("k_lens=[17,9] key padding mask", got,
                             reference(qf, kf, vf, k_lens=k_lens),
                             f32tol) else [f"{label}: k_lens"]

        # 8. GQA: Nq divisible by Nk.
        qg = torch.randn(2, 17, 8, 8)
        kg = torch.randn(2, 17, 4, 8)
        vg = torch.randn(2, 17, 4, 8)
        got = mod.flash_attention(qg, kg, vg, dtype=torch.float32, version=2)
        fails += [] if check("GQA Nq=8 Nk=4", got, reference(qg, kg, vg),
                             f32tol) else [f"{label}: gqa"]

        # 9. Refusals, not silently-wrong answers.
        for why, kwargs in (
            ("window_size", dict(window_size=(4, 4))),
            ("causal with lq != lk", dict(causal=True)),
        ):
            try:
                mod.flash_attention(q1, k, v, version=2, **kwargs)
                print(f"  FAIL {why:<52} returned instead of raising")
                fails.append(f"{label}: {why} did not raise")
            except NotImplementedError:
                print(f"  ok   raises NotImplementedError on {why}")

    fails += check_real_shim()

    print()
    if fails:
        print(f"FAILED: {len(fails)}")
        for f in fails:
            print(" -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
