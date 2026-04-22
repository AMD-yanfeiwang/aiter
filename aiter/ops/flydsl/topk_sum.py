
# SPDX-License-Identifier: MIT
"""Triton kernel for small-dim reduction: sum over topk dimension.

Replaces torch.sum(x.view(token_num, topk, model_dim), dim=1, out=out)
with a single-pass kernel, avoiding PyTorch 2-pass reduce when topk > 8.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 4096}, num_warps=16, num_stages=1),
    ],
    key=["token_num", "model_dim", "TOPK"],
)
@triton.jit
def _topk_sum_kernel(
    inp_ptr,
    out_ptr,
    token_num,
    model_dim,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)

    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offs < model_dim

    stride = tl.cast(model_dim, tl.int64)
    base = pid_m * TOPK * stride

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        offs = base + k * stride + col_offs
        vals = tl.load(inp_ptr + offs, mask=col_mask, other=0.0)
        acc += vals.to(tl.float32)

    out_offs = pid_m * stride + col_offs
    tl.store(out_ptr + out_offs, acc.to(tl.bfloat16), mask=col_mask)


def topk_sum(
    target: torch.Tensor,
    token_num: int,
    topk: int,
    model_dim: int,
    out: torch.Tensor,
) -> None:
    """Sum target.view(token_num, topk, model_dim) over dim=1 into out.

    Single-pass Triton kernel with autotune. Avoids PyTorch 2-pass reduce
    when topk > 8.
    """
    def grid(META):
        return (token_num, triton.cdiv(model_dim, META["BLOCK_N"]))
    _topk_sum_kernel[grid](
        target,
        out,
        token_num,
        model_dim,
        TOPK=topk,
    )
