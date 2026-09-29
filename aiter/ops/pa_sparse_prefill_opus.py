# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS-based sparse paged prefill attention for DeepSeek-V4.

Two-region sparse scaled-dot-product attention over a paged prefix source
(``unified_kv``) and a flat per-fwd extend source (``kv``), with a per-head
softmax-denominator sink. The two regions share a single online-softmax
accumulator, making the order region-invariant.

The user-facing entry is :func:`pa_sparse_prefill_opus`, which dispatches on
the running GPU. Both backends live in ``module_mla_v4_prefill_opus``:

* ``gfx950`` -- :func:`pa_sparse_prefill_gfx950_opus_fwd`, kernel compiled
  from source by the JIT.
* ``gfx1250`` -- :func:`pa_sparse_prefill_gfx1250_opus_fwd`, kernel loaded
  from the prebuilt code objects in ``hsa/gfx1250/mla_v4_opus/``.

Constraints common to both:

* Head dim ``D == 512``.
* dtype ``bf16`` or ``fp16`` for Q/K/V/O; ``attn_sink`` is ``fp32``.
  ``gfx1250`` builds only the ``bf16`` variant.
* Every entry in ``kv_indices_prefix`` / ``kv_indices_extend`` must be a
  valid row index into ``unified_kv`` / ``kv`` respectively. Empty CSR rows
  (``kv_indptr[i] == kv_indptr[i+1]``) are allowed.

See ``csrc/include/mla_v4_prefill_opus.h`` for the C++ API.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes

MD_NAME = "module_mla_v4_prefill_opus"

SUPPORTED_ARCHS = ("gfx950", "gfx1250")
# Archs where pa_sparse_prefill_opus takes the output epilogue kwargs
# (inv_rope_positions / inv_rope_freqs / out_scale); also a feature probe.
OUTPUT_EPILOGUE_ARCHS = ("gfx950",)


def _dispatch(gfx: str, op_gfx950, op_gfx1250):
    """Pick the backend for the running GPU.

    gfx950 compiles the kernel from source. gfx1250 instead loads a prebuilt
    code object (``hsa/gfx1250/mla_v4_opus/``), because its kernel needs
    the CoExec scheduler from a custom LLVM build that release images do not
    ship; see ``csrc/py_itfs_cu/mla_v4_prefill_opus_kernels.cu``.
    """
    if gfx == "gfx1250":
        return op_gfx1250
    if gfx == "gfx950":
        return op_gfx950
    raise RuntimeError(f"pa_sparse_prefill_opus supports {SUPPORTED_ARCHS}, got {gfx}")


@compile_ops(MD_NAME, develop=True)
def pa_sparse_prefill_gfx950_opus_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_freqs: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> None: ...


@compile_ops(MD_NAME, develop=True)
def pa_sparse_prefill_gfx1250_opus_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _pa_sparse_prefill_opus_fake(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_freqs: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty_like(q, dtype=q.dtype if out_scale is None else dtypes.fp8)


@torch_compile_guard(
    mutates_args=["out", "out_scale"], gen_fake=_pa_sparse_prefill_opus_fake
)
def pa_sparse_prefill_opus(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_freqs: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention over two KV sources (paged ``unified_kv`` +
    flat per-fwd ``kv``), backed by the OPUS gfx950 HIP kernel.

    The trailing ``out`` keyword is an aiter-only convenience for callers
    that want to reuse a pre-allocated output buffer; pass ``None`` (the
    default) to have one allocated for you.

    Args:
      q:                 ``[T, H, D]`` bf16/fp16 query (T == N tokens).
      unified_kv:        ``[total_pages, D]`` prefix source (paged history).
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into
        ``unified_kv``, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv:                ``[total_tokens, D]`` extend source (current fwd's
        just-computed K).
      kv_indices_extend: ``[total_extend]`` int32 row indices into ``kv``,
        concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, D]`` output buffer; allocated if
        ``None``.

    Optional output epilogue (gfx950, bf16 only), on the fp32 accumulator:
      inv_rope_positions: ``[>=T]`` int64 positions, with
      inv_rope_freqs:    ``[max_pos, 64]`` fp32 ``view_as_real(freqs_cis)``
        rows (cos at ``2k``, sin at ``2k+1``): applies the inverse GPT-J
        (interleaved) RoPE to the trailing 64 lanes of every head.
      out_scale:         ``T * H * D // 128`` contiguous uint8: ``out`` becomes
        fp8 e4m3fn (mxfp8) and ``out_scale`` receives the e8m0 scale of every
        128-lane group, ``ceil_to_ue8m0(max(amax / 448, 1e-10))``, laid out as
        ``[T, H, D // 128]``.

    Returns:
      ``out`` (``[T, H, D]``, same dtype as ``q``, or fp8 with ``out_scale``).
    """
    gfx = get_gfx_runtime()
    fwd = _dispatch(
        gfx, pa_sparse_prefill_gfx950_opus_fwd, pa_sparse_prefill_gfx1250_opus_fwd
    )

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"pa_sparse_prefill_opus expects fp16/bf16 q, got {q.dtype}")
    if gfx == "gfx1250" and q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"the gfx1250 code object only provides the bf16 variant, got {q.dtype}"
        )
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: unified_kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if unified_kv.size(-1) != kv.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: unified_kv={unified_kv.size(-1)}, kv={kv.size(-1)}"
        )

    epilogue = (inv_rope_positions, inv_rope_freqs, out_scale)
    if gfx not in OUTPUT_EPILOGUE_ARCHS and any(t is not None for t in epilogue):
        raise RuntimeError(
            f"the output epilogue supports {OUTPUT_EPILOGUE_ARCHS}, got {gfx}"
        )
    out_dtype = dtypes.fp8 if out_scale is not None else q.dtype
    if out is None:
        out = torch.empty_like(q, dtype=out_dtype)
    elif out.shape != q.shape or out.dtype != out_dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={out_dtype}"
        )

    args = (
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
    )
    if gfx in OUTPUT_EPILOGUE_ARCHS:
        fwd(*args, *epilogue)
    else:
        fwd(*args)
    return out


@compile_ops(MD_NAME, develop=True)
def pa_sparse_prefill_fp8_gfx950_opus_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_freqs: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> None: ...


@compile_ops(MD_NAME, develop=True)
def pa_sparse_prefill_fp8_gfx1250_opus_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _pa_sparse_prefill_fp8_opus_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_freqs: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    dtype = torch.bfloat16 if out_scale is None else dtypes.fp8
    return torch.empty((t, h, 512), dtype=dtype, device=q_nope.device)


@torch_compile_guard(
    mutates_args=["out", "out_scale"], gen_fake=_pa_sparse_prefill_fp8_opus_fake
)
def pa_sparse_prefill_fp8_opus(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_freqs: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention with split fp8 NoPE and bf16 RoPE inputs.

    The trailing ``out`` keyword is an aiter-only convenience for callers that
    want to reuse a pre-allocated output buffer; pass ``None`` (the default) to
    have one allocated for you.

    Args:
      q_nope:            ``[T, H, 512]`` fp8 query without positional encoding.
      q_rope:            ``[T, H, 64]`` bf16 query RoPE encoding part.
      unified_kv_nope:   ``[total_pages, 512]`` fp8 prefix KV NoPE source.
      unified_kv_rope:   ``[total_pages, 64]`` bf16 prefix KV RoPE source.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        sources, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv_nope:           ``[total_tokens, 512]`` fp8 extend KV NoPE source.
      kv_rope:           ``[total_tokens, 64]`` bf16 extend KV RoPE source.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        sources, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the combined QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output buffer; allocated
        if ``None``.
      inv_rope_positions / inv_rope_freqs / out_scale: optional output
        epilogue (gfx950), as in :func:`pa_sparse_prefill_opus`.

    Returns:
      ``out`` (``[T, H, 512]`` bf16, or fp8 with ``out_scale``).
    """
    gfx = get_gfx_runtime()
    fwd = _dispatch(
        gfx,
        pa_sparse_prefill_fp8_gfx950_opus_fwd,
        pa_sparse_prefill_fp8_gfx1250_opus_fwd,
    )

    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")

    epilogue = (inv_rope_positions, inv_rope_freqs, out_scale)
    if gfx not in OUTPUT_EPILOGUE_ARCHS and any(t is not None for t in epilogue):
        raise RuntimeError(
            f"the output epilogue supports {OUTPUT_EPILOGUE_ARCHS}, got {gfx}"
        )
    out_dtype = torch.bfloat16 if out_scale is None else dtypes.fp8
    t, h = q_nope.shape[0], q_nope.shape[1]
    if out is None:
        out = torch.empty((t, h, 512), dtype=out_dtype, device=q_nope.device)
    elif out.shape != (t, h, 512) or out.dtype != out_dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={out_dtype}"
        )

    args = (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
    )
    if gfx in OUTPUT_EPILOGUE_ARCHS:
        fwd(*args, *epilogue)
    else:
        fwd(*args)
    return out


__all__ = [
    "pa_sparse_prefill_fp8_gfx950_opus_fwd",
    "pa_sparse_prefill_fp8_gfx1250_opus_fwd",
    "pa_sparse_prefill_fp8_opus",
    "pa_sparse_prefill_gfx950_opus_fwd",
    "pa_sparse_prefill_gfx1250_opus_fwd",
    "pa_sparse_prefill_opus",
]
