# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-B MXFP4 MOE Triton operation for DWDP (Zero-Copy, Multi-Pointer).

Provides fused_moe_mxfp4_multi_b() that accepts List[Tensor] weight partitions
and dispatches to the multi-B Triton kernel WITHOUT any data copy (no torch.cat).

Zero-copy approach:
  - Pass each partition tensor directly as a separate kernel argument (up to 4)
  - Kernel selects correct pointer via uniform branch on partition_id
  - No memory copy, no concatenation, no pointer offset arithmetic across allocations

Usage from fused_moe.py entry point:
    if isinstance(w1, list):
        return _fused_moe_multi_b_dispatch(hidden_states, w1, w2, ...)
"""

import torch
import triton
import triton.language as tl
from typing import Any, Dict, List, Optional, Tuple
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton._triton_kernels.moe.moe_op_mxfp4_multi_b import (
    _fused_moe_kernel_mxfp4_multi_b,
    get_scaled_dot_format_string,
)
from aiter.ops.triton.utils.types import torch_to_triton_dtype

_LOGGER = AiterTritonLogger()


def _build_expert_mapping(
    b_list: List[torch.Tensor],
    b_mx_scale_list: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build expert-to-partition mapping tensors (ZERO COPY).

    Returns:
        expert_partition_id[global_E]: which partition (0-3)
        expert_local_idx[global_E]: local expert index within partition
    """
    global_E = sum(b.shape[0] for b in b_list)
    expert_partition_id = torch.empty(global_E, dtype=torch.int32, device=device)
    expert_local_idx = torch.empty(global_E, dtype=torch.int64, device=device)

    expert_idx = 0
    for part_idx, b in enumerate(b_list):
        num_experts_in_part = b.shape[0]
        for local_idx in range(num_experts_in_part):
            expert_partition_id[expert_idx] = part_idx
            expert_local_idx[expert_idx] = local_idx
            expert_idx += 1

    return expert_partition_id, expert_local_idx


def fused_moe_mxfp4_multi_b(
    A: torch.Tensor,
    B_list: List[torch.Tensor],
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale_list: List[torch.Tensor],
    A_mx_scale: Optional[torch.Tensor],
    B_mx_scale_list: List[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """Multi-B MXFP4 MOE kernel launcher (ZERO COPY, NO torch.cat).

    Passes each partition tensor directly to the kernel via explicit pointer args.
    Supports up to 8 partitions.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert A_scale is not None
    assert len(B_list) == len(B_scale_list) == len(B_mx_scale_list)
    assert 1 <= len(B_list) <= 8, f"Multi-B supports 1-8 partitions, got {len(B_list)}"

    device = A.device
    num_partitions = len(B_list)

    if A.dtype == torch.uint8:
        assert A_mx_scale is not None, "A_mx_scale required for MXFP4 activations"
        A_mx_scale_strid_m, A_mx_scale_strid_k = A_mx_scale.stride()
    else:
        A_mx_scale_strid_m, A_mx_scale_strid_k = None, None

    # Build expert mapping (tiny tensors, no weight copy)
    expert_partition_id, expert_local_idx = _build_expert_mapping(
        B_list, B_mx_scale_list, device
    )

    # Per-partition scalar scales (just num_partitions floats)
    b_scale_flat = torch.tensor(
        [s.flatten()[0].item() for s in B_scale_list],
        dtype=torch.float32, device=device,
    )

    # Handle fp4x2 dtype: Triton sees these as uint8
    import aiter
    if B_list[0].dtype == aiter.dtypes.fp4x2:
        B_list = [b.view(torch.uint8) for b in B_list]
    if B_mx_scale_list[0].dtype != torch.uint8:
        B_mx_scale_list = [s.view(torch.uint8) for s in B_mx_scale_list]

    # Pad partition lists to 8 (fill with first partition as dummy)
    b_ptrs = list(B_list) + [B_list[0]] * (8 - num_partitions)
    bmx_ptrs = list(B_mx_scale_list) + [B_mx_scale_list[0]] * (8 - num_partitions)

    # Reference shapes (uniform across partitions)
    B_ref = B_list[0]
    B_mx_ref = B_mx_scale_list[0]
    N = B_ref.shape[1]
    K = A.shape[1]

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    A_tl_dtype = torch_to_triton_dtype[A.dtype]
    A_DTYPE_FORMAT = get_scaled_dot_format_string(A_tl_dtype)
    B_tl_dtype = torch_to_triton_dtype[B_ref.dtype]
    B_DTYPE_FORMAT = get_scaled_dot_format_string(B_tl_dtype)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _fused_moe_kernel_mxfp4_multi_b[grid](
        A,
        b_ptrs[0], b_ptrs[1], b_ptrs[2], b_ptrs[3], b_ptrs[4], b_ptrs[5], b_ptrs[6], b_ptrs[7],  # 8 weight pointers
        C,
        A_scale,
        b_scale_flat,
        A_mx_scale,
        bmx_ptrs[0], bmx_ptrs[1], bmx_ptrs[2], bmx_ptrs[3], bmx_ptrs[4], bmx_ptrs[5], bmx_ptrs[6], bmx_ptrs[7],  # 8 mx_scale pointers
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        # Expert mapping
        expert_partition_id,
        expert_local_idx,
        # Dimensions
        N,
        K,
        topk_ids.numel(),
        # Strides (uniform)
        A.stride(0),
        A.stride(1),
        B_ref.stride(0),   # stride_be (expert stride)
        B_ref.stride(2),   # stride_bk
        B_ref.stride(1),   # stride_bn
        C.stride(1) if C.dim() == 3 else C.stride(0),   # stride_cm
        C.stride(2) if C.dim() == 3 else C.stride(1),   # stride_cn
        A_mx_scale_strid_m,
        A_mx_scale_strid_k,
        B_mx_ref.stride(0),  # stride_bmxe
        B_mx_ref.stride(2),  # stride_bmxk
        B_mx_ref.stride(1),  # stride_bmxn
        # Meta
        A_DTYPE_FORMAT=A_DTYPE_FORMAT,
        B_DTYPE_FORMAT=B_DTYPE_FORMAT,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        NUM_XCDS=get_num_xcds(),
        NUM_PARTITIONS=num_partitions,
        **config,
    )
