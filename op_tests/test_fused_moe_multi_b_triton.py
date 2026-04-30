# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for multi-B Triton MOE kernel (DWDP support).
Tests the multi-B Triton kernel against the single-B Triton kernel.
Both are PURE TRITON - no CK kernel dependency.
DeepSeek-R1 model parameters:
  - n_routed_experts = 256
  - num_experts_per_tok (topk) = 8
  - hidden_size = 7168
  - moe_intermediate_size = 2048
  - n_shared_experts = 1
"""
import argparse
import gc
import sys
import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk
from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
from aiter.ops.triton.moe.moe_op_mxfp4_multi_b import (
    fused_moe_mxfp4_multi_b,
    _build_expert_mapping,
)
from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
import triton.language as tl
torch.set_default_device("cuda")
SEP = "=" * 70
def _split_tensor(t, splits, dim=0):
    parts = torch.split(t, splits, dim=dim)
    return [p.clone().contiguous() for p in parts]
def _check_result(ref_out, test_out, label, atol=0.001, rtol=0.001, pass_pct=99.0):
    ref_f = ref_out.float()
    test_f = test_out.float()
    max_delta = (ref_f - test_f).abs().max().item()
    close_mask = torch.isclose(ref_f, test_f, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close >= pass_pct
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}: max_delta={max_delta:.4f}, "
          f"{pct_close:.1f}% close (atol={atol}, rtol={rtol})")
    if not passed:
        print(f"    ref  sample: {ref_f.reshape(-1)[:8]}")
        print(f"    test sample: {test_f.reshape(-1)[:8]}")
        delta = (ref_f - test_f).abs()
        print(f"    mean_delta={delta.mean().item():.4f}")
    return passed, max_delta, pct_close
def _build_split_configs(E):
    """Build balanced expert partition configs for DWDP multi-B kernel.
    In production DeepSeek-R1: fused_moe only handles 256 routed experts
    (shared expert is computed separately). Multi-B partitions are always
    balanced: 256/1, 256/2, 256/4, 256/8 across GPU ranks.
    """
    configs = []
    # 1-way: single partition (baseline, no split)
    configs.append([E])
    if E >= 2 and E % 2 == 0:
        configs.append([E // 2] * 2)
    if E >= 4 and E % 4 == 0:
        configs.append([E // 4] * 4)
    if E >= 8 and E % 8 == 0:
        configs.append([E // 8] * 8)
    return configs
def _prepare_sorting(M, E, topk, block_size, topk_ids, device="cuda"):
    num_valid_tokens = M * topk
    max_num_tokens_padded = num_valid_tokens + E * block_size
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), num_valid_tokens,
        dtype=torch.int32, device=device,
    )
    max_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids_sorted = torch.full(
        (max_blocks,), -1, dtype=torch.int32, device=device,
    )
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    moe_align_block_size_triton(
        topk_ids, E, block_size,
        sorted_token_ids, expert_ids_sorted, num_tokens_post_pad,
    )
    return sorted_token_ids, expert_ids_sorted, num_tokens_post_pad
def _flat_to_packed_sorted_ids(sorted_token_ids, topk, M):
    """Convert flat sorted_ids (from moe_align_block_size_triton) to packed format.
    Flat format: each entry = token_idx * topk + topk_rank (padding = M*topk)
    Packed format: each entry = (topk_rank << 24) | token_id (as used by moe_sorting_fwd)
    """
    num_valid = M * topk
    flat = sorted_token_ids.clone()
    valid_mask = flat < num_valid
    token_id = torch.where(valid_mask, flat // topk, torch.full_like(flat, M))
    topk_rank = torch.where(valid_mask, flat % topk, torch.full_like(flat, topk))
    packed = (topk_rank << 24) | token_id
    return packed
def test_kernel_multi_b_gemm(M, E, N, K, topk, splits, dtype=torch.bfloat16):
    """Test multi-B kernel vs single-B Triton kernel. Should be BIT-EXACT."""
    label = f"kernel_gemm_M{M}_E{E}_N{N}_K{K}_splits{splits}"
    print(f"\n--- {label} ---")
    torch.manual_seed(42)
    K_packed = K // 2
    K_scale = K // 32
    w_full = torch.randint(0, 256, (E, N, K_packed), dtype=torch.uint8, device="cuda")
    w_mx_full = torch.randint(100, 140, (E, N, K_scale), dtype=torch.uint8, device="cuda")
    a = torch.randint(0, 256, (M, K_packed), dtype=torch.uint8, device="cuda")
    a_mx_scale = torch.randint(100, 140, (M, K_scale), dtype=torch.uint8, device="cuda")
    a_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    b_scale = torch.ones(E, dtype=torch.float32, device="cuda")
    score = torch.randn((M, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(
        torch.randn(M, K, dtype=dtype, device="cuda"), score, topk, True
    )
    block_size = 32
    sorted_token_ids, expert_ids_sorted, num_tokens_post_pad = \
        _prepare_sorting(M, E, topk, block_size, topk_ids)
    config = {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
    }
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    # Reference: single-B Triton kernel
    C_ref = torch.zeros((M, topk, N), dtype=dtype, device="cuda")
    fused_moe_mxfp4(
        A=a, B=w_full, C=C_ref,
        A_scale=a_scale, B_scale=b_scale,
        A_mx_scale=a_mx_scale, B_mx_scale=w_mx_full,
        topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids_sorted,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=True, top_k=topk,
        swizzle_mx_a=False, swizzle_mx_b=False,
        config=config, compute_type=compute_type,
    )
    # Test: multi-B Triton kernel
    C_test = torch.zeros((M, topk, N), dtype=dtype, device="cuda")
    w_list = _split_tensor(w_full, splits, dim=0)
    w_mx_list = _split_tensor(w_mx_full, splits, dim=0)
    b_scale_list = _split_tensor(b_scale.unsqueeze(-1).unsqueeze(-1), splits, dim=0)
    b_scale_list = [s.reshape(-1) for s in b_scale_list]
    fused_moe_mxfp4_multi_b(
        A=a, B_list=w_list, C=C_test,
        A_scale=a_scale, B_scale_list=b_scale_list,
        A_mx_scale=a_mx_scale, B_mx_scale_list=w_mx_list,
        topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids_sorted,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=True, top_k=topk,
        config=config, compute_type=compute_type,
    )
    return _check_result(C_ref, C_test, label, atol=0.001, rtol=0.001, pass_pct=99.0)
def test_full_pipeline_multi_b(M, model_dim, inter_dim, E, topk, splits, dtype=torch.bfloat16):
    """Full 2-stage pipeline: multi-B Triton vs single-B Triton. PURE TRITON."""
    label = f"pipeline_M{M}_E{E}_d{model_dim}_i{inter_dim}_splits{splits}"
    print(f"\n--- {label} ---")
    torch.manual_seed(42)
    K1_packed = model_dim // 2
    K2_packed = inter_dim // 2
    K1_scale = model_dim // 32
    K2_scale = inter_dim // 32
    N1 = inter_dim * 2
    N2 = model_dim
    w1_full = torch.randint(0, 256, (E, N1, K1_packed), dtype=torch.uint8, device="cuda")
    w1_mx_full = torch.randint(100, 140, (E, N1, K1_scale), dtype=torch.uint8, device="cuda")
    w2_full = torch.randint(0, 256, (E, N2, K2_packed), dtype=torch.uint8, device="cuda")
    w2_mx_full = torch.randint(100, 140, (E, N2, K2_scale), dtype=torch.uint8, device="cuda")
    b1_scale = torch.ones(E, dtype=torch.float32, device="cuda")
    b2_scale = torch.ones(E, dtype=torch.float32, device="cuda")
    a_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    x = torch.randn((M, model_dim), dtype=dtype, device="cuda") / 10
    score = torch.randn((M, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, topk, True)
    block_size = 32
    sorted_token_ids, expert_ids_sorted, num_tokens_post_pad = \
        _prepare_sorting(M, E, topk, block_size, topk_ids)
    config = {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 0,
        "waves_per_eu": 0,
    }
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    # Convert flat sorted_ids to packed format for quant kernel
    sorted_token_ids_packed = _flat_to_packed_sorted_ids(sorted_token_ids, topk, M)
    # Quantize activations
    a1_raw, a1_mx_raw = fused_dynamic_mxfp4_quant_moe_sort(
        x, sorted_ids=sorted_token_ids_packed, num_valid_ids=num_tokens_post_pad,
        token_num=M, topk=topk, block_size=block_size,
    )
    # View as uint8 for Triton kernels
    a1 = a1_raw.view(torch.uint8)
    a1_mx_scale = a1_mx_raw.view(torch.uint8)
    # Stage 1: single-B reference
    C1_ref = torch.zeros((M, topk, N1), dtype=dtype, device="cuda")
    fused_moe_mxfp4(
        A=a1, B=w1_full, C=C1_ref,
        A_scale=a_scale, B_scale=b1_scale,
        A_mx_scale=a1_mx_scale, B_mx_scale=w1_mx_full,
        topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids_sorted,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=False, top_k=topk,
        swizzle_mx_a=False, swizzle_mx_b=False,
        config=config, compute_type=compute_type,
    )
    # Stage 1: multi-B test
    C1_test = torch.zeros((M, topk, N1), dtype=dtype, device="cuda")
    w1_list = _split_tensor(w1_full, splits, dim=0)
    w1_mx_list = _split_tensor(w1_mx_full, splits, dim=0)
    b1_scale_list = _split_tensor(b1_scale.unsqueeze(-1).unsqueeze(-1), splits, dim=0)
    b1_scale_list = [s.reshape(-1) for s in b1_scale_list]
    fused_moe_mxfp4_multi_b(
        A=a1, B_list=w1_list, C=C1_test,
        A_scale=a_scale, B_scale_list=b1_scale_list,
        A_mx_scale=a1_mx_scale, B_mx_scale_list=w1_mx_list,
        topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids_sorted,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=False, top_k=topk,
        config=config, compute_type=compute_type,
    )
    # SwiGLU activation
    gate_ref = C1_ref[:, :, :N1 // 2]
    up_ref = C1_ref[:, :, N1 // 2:]
    intermediate_ref = F.silu(gate_ref) * up_ref
    gate_test = C1_test[:, :, :N1 // 2]
    up_test = C1_test[:, :, N1 // 2:]
    intermediate_test = F.silu(gate_test) * up_test
    # Re-quantize intermediate
    a2_ref_raw, a2_mx_ref_raw = fused_dynamic_mxfp4_quant_moe_sort(
        intermediate_ref.reshape(-1, inter_dim),
        sorted_ids=sorted_token_ids_packed, num_valid_ids=num_tokens_post_pad,
        token_num=M, topk=topk, block_size=block_size,
    )
    a2_ref = a2_ref_raw.view(torch.uint8)
    a2_mx_ref = a2_mx_ref_raw.view(torch.uint8)
    a2_test_raw, a2_mx_test_raw = fused_dynamic_mxfp4_quant_moe_sort(
        intermediate_test.reshape(-1, inter_dim),
        sorted_ids=sorted_token_ids_packed, num_valid_ids=num_tokens_post_pad,
        token_num=M, topk=topk, block_size=block_size,
    )
    a2_test = a2_test_raw.view(torch.uint8)
    a2_mx_test = a2_mx_test_raw.view(torch.uint8)
    # Stage 2: single-B reference
    C2_ref = torch.zeros((M, topk, N2), dtype=dtype, device="cuda")
    fused_moe_mxfp4(
        A=a2_ref, B=w2_full, C=C2_ref,
        A_scale=a_scale, B_scale=b2_scale,
        A_mx_scale=a2_mx_ref, B_mx_scale=w2_mx_full,
        topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids_sorted,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=True, top_k=topk,
        swizzle_mx_a=False, swizzle_mx_b=False,
        config=config, compute_type=compute_type,
    )
    # Stage 2: multi-B test
    C2_test = torch.zeros((M, topk, N2), dtype=dtype, device="cuda")
    w2_list = _split_tensor(w2_full, splits, dim=0)
    w2_mx_list = _split_tensor(w2_mx_full, splits, dim=0)
    b2_scale_list = _split_tensor(b2_scale.unsqueeze(-1).unsqueeze(-1), splits, dim=0)
    b2_scale_list = [s.reshape(-1) for s in b2_scale_list]
    fused_moe_mxfp4_multi_b(
        A=a2_test, B_list=w2_list, C=C2_test,
        A_scale=a_scale, B_scale_list=b2_scale_list,
        A_mx_scale=a2_mx_test, B_mx_scale_list=w2_mx_list,
        topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids_sorted,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=True, top_k=topk,
        config=config, compute_type=compute_type,
    )
    out_ref = C2_ref.sum(dim=1)
    out_test = C2_test.sum(dim=1)
    return _check_result(out_ref, out_test, label, atol=0.01, rtol=0.01, pass_pct=99.0)
def test_expert_mapping():
    """Test _build_expert_mapping correctness."""
    print("\n--- test_expert_mapping ---")
    E_parts = [4, 3, 5]
    N, K = 128, 64
    device = "cuda"
    b_list = [torch.randn(e, N, K, device=device) for e in E_parts]
    bmx_list = [torch.randn(e, N, K // 32, device=device) for e in E_parts]
    partition_id, local_idx = _build_expert_mapping(b_list, bmx_list, device)
    expected_partition = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert partition_id.tolist() == expected_partition
    expected_local = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4]
    assert local_idx.tolist() == expected_local
    # DeepSeek-R1: 256 experts, 4-way split (64 per rank)
    b_ds = [torch.randn(64, 128, 32, device=device) for _ in range(4)]
    bmx_ds = [torch.randn(64, 128, 1, device=device) for _ in range(4)]
    pid, lidx = _build_expert_mapping(b_ds, bmx_ds, device)
    assert pid.shape[0] == 256
    assert pid[-1].item() == 3
    assert lidx[-1].item() == 63
    # 256 experts, 2-way uneven (85 + 171)
    b_uneven = [torch.randn(85, 128, 32, device=device), torch.randn(171, 128, 32, device=device)]
    bmx_uneven = [torch.randn(85, 128, 1, device=device), torch.randn(171, 128, 1, device=device)]
    pid2, lidx2 = _build_expert_mapping(b_uneven, bmx_uneven, device)
    assert pid2.shape[0] == 256
    assert pid2[84].item() == 0
    assert pid2[85].item() == 1
    assert lidx2[255].item() == 170
    print("  [PASS] Expert mapping zero-copy offsets verified")
    return True, 0.0, 100.0
def test_dispatch_entry(M, model_dim, inter_dim, E, topk, splits):
    """Test fused_moe() dispatch: N-way split vs 1-way split (precision comparison)."""
    from aiter.fused_moe import fused_moe
    label = f"dispatch_M{M}_E{E}_splits{splits}"
    print(f"\n--- {label} ---")
    torch.manual_seed(42)
    K1_packed = model_dim // 2
    K1_scale = model_dim // 32
    K2_packed = inter_dim // 2
    K2_scale = inter_dim // 32
    N1 = inter_dim * 2
    N2 = model_dim
    w1_full = torch.randint(0, 256, (E, N1, K1_packed), dtype=torch.uint8, device="cuda")
    w1_mx_full = torch.randint(100, 140, (E, N1, K1_scale), dtype=torch.uint8, device="cuda")
    w2_full = torch.randint(0, 256, (E, N2, K2_packed), dtype=torch.uint8, device="cuda")
    w2_mx_full = torch.randint(100, 140, (E, N2, K2_scale), dtype=torch.uint8, device="cuda")
    x = torch.randn((M, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    score = torch.randn((M, E), dtype=torch.bfloat16, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, topk, True)
    w1_list = _split_tensor(w1_full, splits, dim=0)
    w1_mx_list = _split_tensor(w1_mx_full, splits, dim=0)
    w2_list = _split_tensor(w2_full, splits, dim=0)
    w2_mx_list = _split_tensor(w2_mx_full, splits, dim=0)
    try:
        # Test: N-way split dispatch
        out_test = fused_moe(
            x, w1_list, w2_list,
            topk_weights, topk_ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            w1_scale=w1_mx_list,
            w2_scale=w2_mx_list,
        )
        # Reference: 1-way (single partition, same Triton path)
        out_ref = fused_moe(
            x, [w1_full], [w2_full],
            topk_weights, topk_ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            w1_scale=[w1_mx_full],
            w2_scale=[w2_mx_full],
        )
        return _check_result(out_ref, out_test, label, atol=0.001, rtol=0.001, pass_pct=99.0)
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        import traceback; traceback.print_exc()
        return False, 0.0, 0.0
# DeepSeek-R1: n_routed_experts=256, topk=8, hidden=7168, moe_inter=2048
MODEL_CONFIGS = {
    "tiny": dict(model_dim=128, inter_dim=64, E=8, topk=2),
    "small": dict(model_dim=256, inter_dim=128, E=16, topk=4),
    "medium": dict(model_dim=1024, inter_dim=512, E=32, topk=8),
    "deepseek": dict(model_dim=7168, inter_dim=2048, E=32, topk=8),
    "deepseek_e": dict(model_dim=7168, inter_dim=2048, E=256, topk=8),
}
def main():
    parser = argparse.ArgumentParser(description="Multi-B Triton MOE kernel tests (PURE TRITON)")
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 8192])
    parser.add_argument("--config", choices=list(MODEL_CONFIGS.keys()), default="deepseek_e")
    parser.add_argument("--splits", type=int, default=None, help="Only test N-way splits")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-dispatch", action="store_true")
    args = parser.parse_args()
    cfg = MODEL_CONFIGS[args.config]
    E = cfg["E"]
    split_configs = _build_split_configs(E)
    if args.splits:
        split_configs = [s for s in split_configs if len(s) == args.splits]
    print(SEP)
    print(f"Multi-B Triton MOE Tests (PURE TRITON) -- config={args.config}")
    md = cfg['model_dim']
    idd = cfg['inter_dim']
    tk = cfg['topk']
    print(f"  E={E}, model_dim={md}, inter_dim={idd}, topk={tk}")
    print(f"  Tokens: {args.tokens}")
    print(f"  Splits: {split_configs}")
    print(SEP)
    results = []
    # Test 0: Expert mapping
    try:
        p, _, _ = test_expert_mapping()
        results.append(("expert_mapping", p))
    except Exception as e:
        print(f"  [ERROR] expert_mapping: {e}")
        import traceback; traceback.print_exc()
        results.append(("expert_mapping", False))
    # Test 1: Kernel-level GEMM
    for M in args.tokens:
        for splits in split_configs:
            try:
                p, _, _ = test_kernel_multi_b_gemm(
                    M=M, E=E, N=cfg["inter_dim"] * 2,
                    K=cfg["model_dim"], topk=cfg["topk"],
                    splits=splits,
                )
                results.append((f"kernel_M{M}_{len(splits)}way", p))
            except Exception as e:
                print(f"  [ERROR] kernel_M{M}_{len(splits)}way: {e}")
                import traceback; traceback.print_exc()
                results.append((f"kernel_M{M}_{len(splits)}way", False))
            gc.collect()
            torch.cuda.empty_cache()
    # Test 2: Full pipeline (pure Triton)
    if not args.skip_pipeline:
        for M in args.tokens:
            for splits in split_configs:
                try:
                    p, _, _ = test_full_pipeline_multi_b(M=M, splits=splits, **cfg)
                    results.append((f"pipeline_M{M}_{len(splits)}way", p))
                except Exception as e:
                    print(f"  [ERROR] pipeline_M{M}_{len(splits)}way: {e}")
                    import traceback; traceback.print_exc()
                    results.append((f"pipeline_M{M}_{len(splits)}way", False))
                gc.collect()
                torch.cuda.empty_cache()
    # Test 3: Dispatch entry point
    if not args.skip_dispatch:
        for M in args.tokens:
            for splits in split_configs:
                try:
                    p, _, _ = test_dispatch_entry(M=M, splits=splits, **cfg)
                    results.append((f"dispatch_M{M}_{len(splits)}way", p))
                except Exception as e:
                    print(f"  [ERROR] dispatch_M{M}_{len(splits)}way: {e}")
                    import traceback; traceback.print_exc()
                    results.append((f"dispatch_M{M}_{len(splits)}way", False))
    # Summary
    print()
    print(SEP)
    print("SUMMARY")
    print(SEP)
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    for label, passed in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {label}")
    print(f"  {n_pass}/{n_total} passed")
    print(SEP)
    return 0 if n_pass == n_total else 1
if __name__ == "__main__":
    sys.exit(main())
