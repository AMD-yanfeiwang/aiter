# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-B MXFP4 MOE Triton kernel for DWDP (Zero-Copy, Multi-Pointer).

Supports up to 8 weight partitions via explicit per-partition base pointers.
The kernel uses scalar branching (uniform across block) to select the correct
partition pointer for each expert. This avoids torch.cat entirely and works
with independently-allocated tensors (non-contiguous in virtual address space).

Key design:
  - 4 weight base pointers (b_ptr_0..3) and 4 mx_scale base pointers
  - expert_partition_id[E] -> which partition (0-3)
  - expert_local_idx[E] -> local expert index within partition
  - Kernel branches on partition_id (uniform, no warp divergence since all
    threads in a block process the same expert)
"""

import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils._triton.moe_common import _write_zeros_to_output
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


_fused_moe_kernel_mxfp4_multi_b_repr = make_kernel_repr(
    "_fused_moe_kernel_mxfp4_multi_b",
    [
        "A_DTYPE_FORMAT",
        "B_DTYPE_FORMAT",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "MUL_ROUTED_WEIGHT",
        "top_k",
        "compute_type",
        "NUM_XCDS",
        "NUM_PARTITIONS",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_moe_kernel_mxfp4_multi_b_repr)
def _fused_moe_kernel_mxfp4_multi_b(
    # Pointers to matrices
    a_ptr,
    b_ptr_0, b_ptr_1, b_ptr_2, b_ptr_3, b_ptr_4, b_ptr_5, b_ptr_6, b_ptr_7,  # up to 8 weight partition pointers
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,     # [num_partitions] per-partition scale
    a_mx_scale_ptr,
    bmx_ptr_0, bmx_ptr_1, bmx_ptr_2, bmx_ptr_3, bmx_ptr_4, bmx_ptr_5, bmx_ptr_6, bmx_ptr_7,  # up to 8 mx_scale partition pointers
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Expert-to-partition mapping
    expert_partition_id_ptr,  # [global_E] -> partition index (0-3)
    expert_local_idx_ptr,     # [global_E] -> local expert index within partition
    # Matrix dimensions
    N,
    K,
    num_valid_tokens,
    # Strides (uniform across partitions)
    stride_am,
    stride_ak,
    stride_be,       # expert stride within partition
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_amxm,
    stride_amxk,
    stride_bmxe,     # expert stride in mx_scale
    stride_bmxk,
    stride_bmxn,
    # Meta-parameters
    A_DTYPE_FORMAT: tl.constexpr,
    B_DTYPE_FORMAT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    """Multi-B MXFP4 MOE kernel with explicit partition pointers."""
    is_a_microscaled_format: tl.constexpr = a_mx_scale_ptr is not None
    is_b_microscaled_format: tl.constexpr = bmx_ptr_0 is not None
    MX_PACK_DIVISOR: tl.constexpr = 32

    if is_a_microscaled_format:
        a_type: tl.constexpr = a_ptr.dtype.element_ty
        tl.static_assert(
            a_type == tl.uint8 or (a_type == tl.float8e4nv or a_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            a_mx_scale_ptr.dtype.element_ty == tl.uint8, "a_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )
    if is_b_microscaled_format:
        b_type: tl.constexpr = b_ptr_0.dtype.element_ty
        tl.static_assert(
            b_type == tl.uint8 or (b_type == tl.float8e4nv or b_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            bmx_ptr_0.dtype.element_ty == tl.uint8, "bmx_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )

    # -----------------------------------------------------------
    # Map program ids to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # ----------------------------------------------------------
    # Load token info
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    # Clamp padding entries to 0 to avoid OOB pointer computation on AMD GPUs
    # (masked loads/stores can still fault on unmapped addresses)
    offs_token = tl.where(token_mask, offs_token, tl.zeros_like(offs_token))

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        _write_zeros_to_output(
            c_ptr, stride_cm, stride_cn, pid_n, N,
            offs_token, token_mask,
            BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type,
        )
        return

    # ----------------------------------------------------------
    # Multi-B: select partition pointer for this expert
    partition_id = tl.load(expert_partition_id_ptr + off_expert).to(tl.int32)
    local_expert_idx = tl.load(expert_local_idx_ptr + off_expert).to(tl.int64)
    bscale_idx = partition_id.to(tl.int64)

    # Select weight base pointer (uniform branch - all threads same expert)
    if NUM_PARTITIONS == 1:
        b_base = b_ptr_0 + local_expert_idx * stride_be
    else:
        if partition_id == 0:
            b_base = b_ptr_0 + local_expert_idx * stride_be
        elif partition_id == 1:
            b_base = b_ptr_1 + local_expert_idx * stride_be
        elif partition_id == 2:
            b_base = b_ptr_2 + local_expert_idx * stride_be
        elif partition_id == 3:
            b_base = b_ptr_3 + local_expert_idx * stride_be
        elif partition_id == 4:
            b_base = b_ptr_4 + local_expert_idx * stride_be
        elif partition_id == 5:
            b_base = b_ptr_5 + local_expert_idx * stride_be
        elif partition_id == 6:
            b_base = b_ptr_6 + local_expert_idx * stride_be
        else:
            b_base = b_ptr_7 + local_expert_idx * stride_be

    # Load scales
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + bscale_idx)

    # Set offsets of B on dim N
    offs_b_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_b_n = tl.max_contiguous(
        tl.multiple_of(offs_b_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N
    )

    # Setup A microscale pointers
    if is_a_microscaled_format:
        A_PACK_DIVISOR: tl.constexpr = 2 if a_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // A_PACK_DIVISOR
        MX_SCALE_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR

        offs_scale_ak = tl.arange(0, MX_SCALE_BLOCK_K_A)
        offs_scale_m = offs_token
        a_mx_scale_ptrs = (
            a_mx_scale_ptr
            + offs_scale_ak.to(tl.int64)[None, :] * stride_amxk
            + offs_scale_m.to(tl.int64)[:, None] // top_k * stride_amxm
        )
    else:
        a_mx_scale_ptrs = None
        A_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_A: tl.constexpr = 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K

    # Setup B microscale pointers - select partition
    if is_b_microscaled_format:
        B_PACK_DIVISOR: tl.constexpr = 2 if b_ptr_0.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // B_PACK_DIVISOR
        MX_SCALE_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR

        if NUM_PARTITIONS == 1:
            bmx_base = bmx_ptr_0 + local_expert_idx * stride_bmxe
        else:
            if partition_id == 0:
                bmx_base = bmx_ptr_0 + local_expert_idx * stride_bmxe
            elif partition_id == 1:
                bmx_base = bmx_ptr_1 + local_expert_idx * stride_bmxe
            elif partition_id == 2:
                bmx_base = bmx_ptr_2 + local_expert_idx * stride_bmxe
            elif partition_id == 3:
                bmx_base = bmx_ptr_3 + local_expert_idx * stride_bmxe
            elif partition_id == 4:
                bmx_base = bmx_ptr_4 + local_expert_idx * stride_bmxe
            elif partition_id == 5:
                bmx_base = bmx_ptr_5 + local_expert_idx * stride_bmxe
            elif partition_id == 6:
                bmx_base = bmx_ptr_6 + local_expert_idx * stride_bmxe
            else:
                bmx_base = bmx_ptr_7 + local_expert_idx * stride_bmxe

        offs_scale_bk = tl.arange(0, MX_SCALE_BLOCK_K_B)
        offs_scale_n = offs_b_n
        b_mx_scale_ptrs = (
            bmx_base
            + offs_scale_bk.to(tl.int64)[None, :] * stride_bmxk
            + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
        )
    else:
        b_mx_scale_ptrs = None
        B_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_B: tl.constexpr = 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K

    # Setup A and B data pointers
    offs_a_k = tl.arange(0, PACKED_BLOCK_K_A)
    offs_b_k = tl.arange(0, PACKED_BLOCK_K_B)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_a_k[None, :] * stride_ak
    )
    b_ptrs = b_base + (offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Main GEMM loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, PACKED_BLOCK_K_A)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None]
                & (offs_a_k[None, :] < (K - k * PACKED_BLOCK_K_A)),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_b_k[:, None] < (K - k * PACKED_BLOCK_K_B),
                other=0.0,
            )

        if is_a_microscaled_format or is_b_microscaled_format:
            if is_a_microscaled_format:
                mask_ak_scale = offs_scale_ak < (K - k * PACKED_BLOCK_K_A) // (
                    MX_PACK_DIVISOR // A_PACK_DIVISOR
                )
                a_mx_scales = tl.load(
                    a_mx_scale_ptrs, mask=mask_ak_scale[None, :], other=0.0
                )
            else:
                a_mx_scales = None

            mask_bk_scale = offs_scale_bk < (K - k * PACKED_BLOCK_K_B) // (
                MX_PACK_DIVISOR // B_PACK_DIVISOR
            )
            b_mx_scales = tl.load(
                b_mx_scale_ptrs, mask=mask_bk_scale[None, :], other=0.0
            )

            accumulator = tl.dot_scaled(
                a, a_mx_scales, A_DTYPE_FORMAT,
                b, b_mx_scales, B_DTYPE_FORMAT,
                acc=accumulator, fast_math=True,
            )

            if is_a_microscaled_format:
                a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A * stride_amxk
            b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B * stride_bmxk

        a_ptrs += PACKED_BLOCK_K_A * stride_ak
        b_ptrs += PACKED_BLOCK_K_B * stride_bk

    # Scale and weight
    accumulator *= a_scale * b_scale
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    # Write output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
