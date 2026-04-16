# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for multi-B weight tensor (DWDP) support in FlyDSL MOE kernels.

Tests both FP8 and FP4 precision with DeepSeek-R1-0528 model configuration
(E=256, model_dim=7168, inter_dim=2048, topk=8).

**IMPORTANT: 4 GB buffer address limit**
AMD buffer_load instructions use i32 byte offsets, limiting a single weight
tensor to < 4 GB.  For E=256, the w1 tensor is ~7 GB (FP8) which overflows.
Multi-B splits keep each partition under 4 GB, resolving this by design.
The single-tensor FP8 test for stage1 is expected to show degraded accuracy
when weight tensor exceeds 4 GB.

Tests per precision:
  1. Backward compatibility: single tensor
  2. Two-way even split
  3. Two-way uneven split
  4. Three-way split
  5. Four-way split

Usage:
    python test_flydsl_moe_multi_b.py                          # all tests
    python test_flydsl_moe_multi_b.py --precision fp8          # FP8 only
    python test_flydsl_moe_multi_b.py --precision fp4          # FP4 only
    python test_flydsl_moe_multi_b.py --stage stage1           # stage1 only
    python test_flydsl_moe_multi_b.py --stage stage2           # stage2 only
    python test_flydsl_moe_multi_b.py -t 64                    # specific token count
"""

import argparse
import sys
import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort

torch.set_default_device("cuda")

SENTINEL = 2**30


# ============================================================================
# Shared helpers
# ============================================================================


def _split_weights(w_shuf, w_scale, splits):
    """Split weight tensor along expert dim (dim 0) into partitions.

    Handles both FP8 (scale dim0 == E) and FP4 (scale dim0 == E*N) layouts.

    Each partition is .clone()'d to get an independent memory allocation.
    This avoids a test coverage gap: torch.split() returns views sharing
    the same underlying storage, so an out-of-bounds read from partition 0
    could silently land in partition 1's valid memory instead of faulting.
    """
    assert sum(splits) == w_shuf.shape[0], (
        f"splits {splits} don't sum to {w_shuf.shape[0]}"
    )
    w_list = [t.clone() for t in torch.split(w_shuf, splits, dim=0)]
    if w_scale is not None:
        if w_scale.shape[0] == w_shuf.shape[0]:
            # FP8 case: scale dim0 matches expert count
            ws_list = [t.clone() for t in torch.split(w_scale, splits, dim=0)]
        else:
            # FP4 case: scale dim0 = E * rows_per_expert (e.g., E*N for e8m0_shuffle)
            total_experts = sum(splits)
            rows_per_expert = w_scale.shape[0] // total_experts
            scale_splits = [s * rows_per_expert for s in splits]
            ws_list = [t.clone() for t in torch.split(w_scale, scale_splits, dim=0)]
    else:
        ws_list = [None] * len(splits)
    offsets = [0]
    for s in splits[:-1]:
        offsets.append(offsets[-1] + s)
    offsets += [SENTINEL] * (5 - len(offsets))
    return w_list, ws_list, tuple(offsets)


def _check_result(ref_out, test_out, label, atol=0.5, rtol=0.1, pass_pct=95.0):
    """Compare outputs and return (passed, max_delta, pct_close)."""
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    close_mask = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}: max_delta={max_delta:.4f}, {pct_close:.1f}% close "
          f"(atol={atol}, rtol={rtol})")
    if not passed:
        print(f"    ref  sample: {ref_out.reshape(-1)[:8]}")
        print(f"    test sample: {test_out.reshape(-1)[:8]}")
    return passed, max_delta, pct_close


def _build_split_configs(E):
    """Build split configurations for a given number of experts."""
    configs = []
    if E >= 2 and E % 2 == 0:
        configs.append([E // 2, E // 2])
    if E >= 3:
        configs.append([E // 2, E - E // 2])
    if E >= 6:
        q = E // 3
        configs.append([q, q, E - 2 * q])
    if E >= 8:
        q = E // 4
        configs.append([q, q, q, E - 3 * q])
    return configs


def _tensor_size_gb(shape, elem_bytes=1):
    """Compute tensor size in GB."""
    import math
    n = math.prod(shape)
    return n * elem_bytes / (1024**3)


# ============================================================================
# FP8 data generation and tests
# ============================================================================


def _generate_fp8_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int = 32,
    dtype=torch.float16,
):
    """Generate FP8 quantized data for testing."""
    Q_TYPE = QuantType.per_Token
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    # FP8 quantize weights
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp8)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp8)

    # FP8 quantize activations
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=dtypes.fp8)

    # Reference stage1 (doweight=False for stage1)
    ref1 = torch_moe_stage1(
        a1_qt, w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, quant_type=Q_TYPE,
        w1_scale=w1_scale, a1_scale=a1_scale,
        doweight=False,
    )

    # Reference stage2 (doweight=True for stage2)
    a2_qt, a2_scale = torch_quant(ref1, quant_dtype=dtypes.fp8)
    ref2 = torch_moe_stage2(
        a2_qt, w1_qt, w2_qt,
        topk_weights, topk_ids,
        dtype=dtype, quant_type=Q_TYPE,
        w2_scale=w2_scale, a2_scale=a2_scale,
        doweight=True,
    )

    # MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    # Preshuffle weights for FlyDSL kernels
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shuf = shuffle_weight(w2_qt, (16, 16))

    return dict(
        a1_qt=a1_qt, a1_scale=a1_scale,
        a2_qt=a2_qt, a2_scale=a2_scale,
        w1_qt_shuf=w1_qt_shuf, w1_scale=w1_scale,
        w2_qt_shuf=w2_qt_shuf, w2_scale=w2_scale,
        sorted_ids=sorted_ids, sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids,
        topk_weights=topk_weights, topk_ids=topk_ids,
        ref_stage1=ref1, ref_stage2=ref2,
        token=token, model_dim=model_dim, inter_dim=inter_dim,
        E=E, topk=topk, dtype=dtype,
    )


def test_fp8_stage1_single(data, tile_m=32, tile_n=256, tile_k=256, **check_kw):
    """FP8 Stage1: single tensor (backward compat)."""
    w1_size = _tensor_size_gb(data["w1_qt_shuf"].shape)
    warn = " [WARN: w1 > 4GB, i32 overflow expected]" if w1_size > 4.0 else ""
    print(f"\n--- FP8 Stage1: Single tensor{warn} ---")
    out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp8", b_dtype="fp8", out_dtype="f16",
        w1_scale=data["w1_scale"],
        a1_scale=data["a1_scale"],
        sorted_weights=None,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    return _check_result(data["ref_stage1"], out, "fp8_stage1_single", **check_kw)


def test_fp8_stage1_multi_b(data, splits, tile_m=32, tile_n=256, tile_k=256, **check_kw):
    """FP8 Stage1: multi-B split."""
    label = f"fp8_stage1_{len(splits)}way_{splits}"
    print(f"\n--- FP8 Stage1: {label} ---")
    w_list, ws_list, offsets = _split_weights(
        data["w1_qt_shuf"], data["w1_scale"], splits
    )
    out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=w_list,
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp8", b_dtype="fp8", out_dtype="f16",
        w1_scale=ws_list,
        a1_scale=data["a1_scale"],
        sorted_weights=None,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        num_b_tensors=len(splits),
        b_tensor_l_offsets=offsets,
    )
    return _check_result(data["ref_stage1"], out, label, **check_kw)


def test_fp8_stage2_single(data, tile_m=32, tile_n=128, tile_k=256, **check_kw):
    """FP8 Stage2: single tensor (backward compat)."""
    print("\n--- FP8 Stage2: Single tensor ---")
    out = flydsl_moe_stage2(
        inter_states=data["a2_qt"],
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp8", b_dtype="fp8", out_dtype="f16",
        w2_scale=data["w2_scale"],
        a2_scale=data["a2_scale"],
        sorted_weights=data["sorted_weights"],  # doweight=True to match ref
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    return _check_result(data["ref_stage2"], out, "fp8_stage2_single", **check_kw)


def test_fp8_stage2_multi_b(data, splits, tile_m=32, tile_n=128, tile_k=256, **check_kw):
    """FP8 Stage2: multi-B split."""
    label = f"fp8_stage2_{len(splits)}way_{splits}"
    print(f"\n--- FP8 Stage2: {label} ---")
    w_list, ws_list, offsets = _split_weights(
        data["w2_qt_shuf"], data["w2_scale"], splits
    )
    out = flydsl_moe_stage2(
        inter_states=data["a2_qt"],
        w2=w_list,
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp8", b_dtype="fp8", out_dtype="f16",
        w2_scale=ws_list,
        a2_scale=data["a2_scale"],
        sorted_weights=data["sorted_weights"],  # doweight=True to match ref
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        num_b_tensors=len(splits),
        b_tensor_l_offsets=offsets,
    )
    return _check_result(data["ref_stage2"], out, label, **check_kw)


# ============================================================================
# FP4 data generation and tests
# ============================================================================


def _generate_fp4_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int = 32,
    dtype=torch.bfloat16,
):
    """Generate FP4 (a4w4, per_1x32) quantized data for testing.

    Memory-efficient: quantizes weights per-expert to avoid the 28 GB fp32
    intermediate that f32_to_mxfp4 creates on the full (E, N, K) tensor.
    """
    import gc
    Q_TYPE = QuantType.per_1x32
    Q_DTYPE_A = dtypes.fp4x2
    Q_DTYPE_W = dtypes.fp4x2
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    del score; gc.collect(); torch.cuda.empty_cache()

    # --- Quantize w1 per-expert to avoid OOM in f32_to_mxfp4 ---
    print("  Generating w1 and quantizing per-expert...")
    CHUNK = 16  # quantize CHUNK experts at a time
    w1_qt_chunks, w1_scale_chunks = [], []
    w1_bf16_chunks = []  # keep bf16 for torch reference
    for start in range(0, E, CHUNK):
        end = min(start + CHUNK, E)
        chunk = torch.randn((end - start, inter_dim * 2, model_dim), dtype=dtype) / 10
        w1_bf16_chunks.append(chunk)
        qt, sc = torch_quant(chunk, quant_dtype=Q_DTYPE_W)
        w1_qt_chunks.append(qt.view(end - start, inter_dim * 2, model_dim // 2))
        w1_scale_chunks.append(sc)
        del qt, sc; torch.cuda.empty_cache()

    w1_qt = torch.cat(w1_qt_chunks, dim=0)
    w1_scale = torch.cat(w1_scale_chunks, dim=0)
    del w1_qt_chunks, w1_scale_chunks

    # --- Quantize w2 per-expert ---
    print("  Generating w2 and quantizing per-expert...")
    w2_qt_chunks, w2_scale_chunks = [], []
    w2_bf16_chunks = []
    for start in range(0, E, CHUNK):
        end = min(start + CHUNK, E)
        chunk = torch.randn((end - start, model_dim, inter_dim), dtype=dtype) / 10
        w2_bf16_chunks.append(chunk)
        qt, sc = torch_quant(chunk, quant_dtype=Q_DTYPE_W)
        w2_qt_chunks.append(qt.view(end - start, model_dim, inter_dim // 2))
        w2_scale_chunks.append(sc)
        del qt, sc; torch.cuda.empty_cache()

    w2_qt = torch.cat(w2_qt_chunks, dim=0)
    w2_scale = torch.cat(w2_scale_chunks, dim=0)
    del w2_qt_chunks, w2_scale_chunks

    # Quantize activation
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=Q_DTYPE_A)

    # Torch reference: stage1 (doweight=False)
    # torch_moe_stage1 needs quantized w1/w2 views
    print("  Computing torch reference stage1...")
    w1_qt_ref = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt_ref = w2_qt.view(E, model_dim, inter_dim // 2)
    ref1 = torch_moe_stage1(
        a1_qt, w1_qt_ref, w2_qt_ref,
        topk_weights, topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=Q_TYPE,
        a1_scale=a1_scale, w1_scale=w1_scale,
        doweight=False,
    )
    gc.collect(); torch.cuda.empty_cache()

    # Torch reference: stage2 (doweight=True)
    print("  Computing torch reference stage2...")
    a2_qt, a2_scale = torch_quant(ref1, quant_dtype=Q_DTYPE_A)
    a2_qt = a2_qt.view(token, topk, -1)
    ref2 = torch_moe_stage2(
        a2_qt, w1_qt_ref, w2_qt_ref,
        topk_weights, topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale, a2_scale=a2_scale,
        doweight=True,
    )
    del w1_qt_ref, w2_qt_ref, w1_bf16_chunks, w2_bf16_chunks
    gc.collect(); torch.cuda.empty_cache()

    # MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    # Preshuffle weights and scales for FlyDSL FP4 kernels
    print("  Shuffling weights for FlyDSL...")
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    del w1_qt; gc.collect(); torch.cuda.empty_cache()
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    del w2_qt; gc.collect(); torch.cuda.empty_cache()
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)

    # Sort activation scales for MoE dispatch
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )
    a2_scale_sort = moe_mxfp4_sort(
        a2_scale[:token * topk, :].view(token, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )

    return dict(
        a1_qt=a1_qt, a1_scale=a1_scale, a1_scale_sort=a1_scale_sort,
        a2_qt=a2_qt, a2_scale=a2_scale, a2_scale_sort=a2_scale_sort,
        w1_qt_shuf=w1_qt_shuf, w1_scale_shuf=w1_scale_shuf,
        w2_qt_shuf=w2_qt_shuf, w2_scale_shuf=w2_scale_shuf,
        w1_scale=w1_scale, w2_scale=w2_scale,
        sorted_ids=sorted_ids, sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids,
        topk_weights=topk_weights, topk_ids=topk_ids,
        ref_stage1=ref1, ref_stage2=ref2,
        token=token, model_dim=model_dim, inter_dim=inter_dim,
        E=E, topk=topk, dtype=dtype,
    )


def test_fp4_stage1_single(data, tile_m=32, tile_n=256, tile_k=256, **check_kw):
    """FP4 Stage1: single tensor (backward compat)."""
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"
    print(f"\n--- FP4 Stage1: Single tensor ---")
    out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp4", b_dtype="fp4", out_dtype=out_dtype_str,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a1_scale_sort"],
        sorted_weights=None,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    return _check_result(data["ref_stage1"], out, "fp4_stage1_single", **check_kw)


def test_fp4_stage1_multi_b(data, splits, tile_m=32, tile_n=256, tile_k=256, **check_kw):
    """FP4 Stage1: multi-B split."""
    label = f"fp4_stage1_{len(splits)}way_{splits}"
    print(f"\n--- FP4 Stage1: {label} ---")
    w_list, ws_list, offsets = _split_weights(
        data["w1_qt_shuf"], data["w1_scale_shuf"], splits
    )
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"
    out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=w_list,
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp4", b_dtype="fp4", out_dtype=out_dtype_str,
        w1_scale=ws_list,
        a1_scale=data["a1_scale_sort"],
        sorted_weights=None,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        num_b_tensors=len(splits),
        b_tensor_l_offsets=offsets,
    )
    return _check_result(data["ref_stage1"], out, label, **check_kw)


def test_fp4_stage2_single(data, tile_m=32, tile_n=256, tile_k=256, **check_kw):
    """FP4 Stage2: single tensor (backward compat)."""
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"
    print(f"\n--- FP4 Stage2: Single tensor ---")
    out = flydsl_moe_stage2(
        inter_states=data["a2_qt"],
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp4", b_dtype="fp4", out_dtype=out_dtype_str,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=data["a2_scale_sort"],
        sorted_weights=data["sorted_weights"],  # doweight=True to match ref
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    return _check_result(data["ref_stage2"], out, "fp4_stage2_single", **check_kw)


def test_fp4_stage2_multi_b(data, splits, tile_m=32, tile_n=256, tile_k=256, **check_kw):
    """FP4 Stage2: multi-B split."""
    label = f"fp4_stage2_{len(splits)}way_{splits}"
    print(f"\n--- FP4 Stage2: {label} ---")
    w_list, ws_list, offsets = _split_weights(
        data["w2_qt_shuf"], data["w2_scale_shuf"], splits
    )
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"
    out = flydsl_moe_stage2(
        inter_states=data["a2_qt"],
        w2=w_list,
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        a_dtype="fp4", b_dtype="fp4", out_dtype=out_dtype_str,
        w2_scale=ws_list,
        a2_scale=data["a2_scale_sort"],
        sorted_weights=data["sorted_weights"],  # doweight=True to match ref
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        num_b_tensors=len(splits),
        b_tensor_l_offsets=offsets,
    )
    return _check_result(data["ref_stage2"], out, label, **check_kw)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-B DWDP MOE tests (FP8 + FP4)")
    parser.add_argument("--stage", choices=["stage1", "stage2", "all"], default="all")
    parser.add_argument("--precision", choices=["fp8", "fp4", "all"], default="all")
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[1, 3, 4, 16, 32, 33, 64, 256, 1024, 2048, 8192])
    parser.add_argument("-E", "--experts", type=int, default=257)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--atol-fp8", type=float, default=0.5)
    parser.add_argument("--rtol-fp8", type=float, default=0.1)
    parser.add_argument("--atol-fp4", type=float, default=1.0)
    parser.add_argument("--rtol-fp4", type=float, default=0.05)
    parser.add_argument("--pass-pct", type=float, default=95.0)
    args = parser.parse_args()

    results = []
    E = args.experts
    split_configs = _build_split_configs(E)

    for token in args.tokens:
        # ==================================================================
        # FP8 tests
        # ==================================================================
        if args.precision in ("fp8", "all"):
            print(f"\n{'='*70}")
            print(f"FP8: token={token}, E={E}, topk={args.topk}, "
                  f"model_dim={args.model_dim}, inter_dim={args.inter_dim}")
            print(f"{'='*70}")

            fp8_check = dict(atol=args.atol_fp8, rtol=args.rtol_fp8, pass_pct=args.pass_pct)

            try:
                fp8_data = _generate_fp8_data(
                    token=token, model_dim=args.model_dim, inter_dim=args.inter_dim,
                    E=E, topk=args.topk, block_m=args.block_m,
                )

                if args.stage in ("stage1", "all"):
                    # Single tensor — may fail if w1 > 4 GB (expected)
                    w1_gb = _tensor_size_gb(fp8_data["w1_qt_shuf"].shape)
                    single_check = dict(fp8_check)
                    if w1_gb > 4.0:
                        # Relax threshold for single tensor with >4GB w1
                        single_check["pass_pct"] = 0.0  # informational only
                    p, _, _ = test_fp8_stage1_single(fp8_data, **single_check)
                    results.append(("fp8_s1_single", p, w1_gb > 4.0))

                    for splits in split_configs:
                        p, _, _ = test_fp8_stage1_multi_b(fp8_data, splits, **fp8_check)
                        results.append((f"fp8_s1_{len(splits)}way", p, False))

                if args.stage in ("stage2", "all"):
                    p, _, _ = test_fp8_stage2_single(fp8_data, **fp8_check)
                    results.append(("fp8_s2_single", p, False))

                    for splits in split_configs:
                        p, _, _ = test_fp8_stage2_multi_b(fp8_data, splits, **fp8_check)
                        results.append((f"fp8_s2_{len(splits)}way", p, False))

            except Exception:
                import traceback
                traceback.print_exc()
                results.append(("fp8_ERROR", False, False))

            # Free FP8 data
            try:
                del fp8_data
            except NameError:
                pass
            import gc; gc.collect(); torch.cuda.empty_cache()

        # ==================================================================
        # FP4 tests
        # ==================================================================
        if args.precision in ("fp4", "all"):
            # Free FP8 data before FP4 to avoid OOM
            if 'fp8_data' in dir():
                del fp8_data
            import gc; gc.collect(); torch.cuda.empty_cache()

            print(f"\n{'='*70}")
            print(f"FP4: token={token}, E={E}, topk={args.topk}, "
                  f"model_dim={args.model_dim}, inter_dim={args.inter_dim}")
            print(f"{'='*70}")

            fp4_check = dict(atol=args.atol_fp4, rtol=args.rtol_fp4, pass_pct=args.pass_pct)

            try:
                fp4_data = _generate_fp4_data(
                    token=token, model_dim=args.model_dim, inter_dim=args.inter_dim,
                    E=E, topk=args.topk, block_m=args.block_m,
                )

                if args.stage in ("stage1", "all"):
                    p, _, _ = test_fp4_stage1_single(fp4_data, **fp4_check)
                    results.append(("fp4_s1_single", p, False))

                    for splits in split_configs:
                        p, _, _ = test_fp4_stage1_multi_b(fp4_data, splits, **fp4_check)
                        results.append((f"fp4_s1_{len(splits)}way", p, False))

                if args.stage in ("stage2", "all"):
                    p, _, _ = test_fp4_stage2_single(fp4_data, **fp4_check)
                    results.append(("fp4_s2_single", p, False))

                    for splits in split_configs:
                        p, _, _ = test_fp4_stage2_multi_b(fp4_data, splits, **fp4_check)
                        results.append((f"fp4_s2_{len(splits)}way", p, False))

            except Exception:
                import traceback
                traceback.print_exc()
                results.append(("fp4_ERROR", False, False))

            # Free FP4 data
            try:
                del fp4_data
            except NameError:
                pass
            import gc; gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    n_pass = 0
    n_total = 0
    n_expected_fail = 0
    for name, passed, expected_fail in results:
        if expected_fail:
            status = "PASS*" if passed else "XFAIL"
            n_expected_fail += 1
            if passed:
                n_pass += 1
            # Don't count expected failures against total
        else:
            status = "PASS" if passed else "FAIL"
            n_total += 1
            if passed:
                n_pass += 1
        print(f"  {status:>5s}  {name}")

    print(f"\n  {n_pass}/{n_total} passed", end="")
    if n_expected_fail > 0:
        print(f" ({n_expected_fail} expected failures due to 4GB limit)", end="")
    print()
    print(f"{'='*70}")

    # Return 0 if all non-expected-failure tests passed
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
