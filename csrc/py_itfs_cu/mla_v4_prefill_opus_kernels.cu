// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based sparse paged prefill attention, one launcher pair per target:
//
//   gfx950  -- kernels compiled from the device templates in
//              `mla_v4_prefill_opus.h` (single-header, IMPL-guarded).
//   gfx1250 -- kernels loaded from the prebuilt code objects in
//              `hsa/gfx1250/mla_v4_opus/`, one per precision and wave layout.

#define OPUS_MLA_V4_PREFILL_IMPL
#include "mla_v4_prefill_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

// Validates the optional output epilogue tensors and fills `epi` from them
// (see opus_mla_v4_prefill_epilogue_args).
void set_epilogue_args(opus_mla_v4_prefill_epilogue_args& epi,
                       const std::optional<aiter_tensor_t>& positions,
                       const std::optional<aiter_tensor_t>& freqs,
                       const std::optional<aiter_tensor_t>& out_scale,
                       int N,
                       int H,
                       int D)
{
    AITER_CHECK(positions.has_value() == freqs.has_value(),
                "inv_rope_positions and inv_rope_freqs go together");
    if(positions)
    {
        AITER_CHECK(positions->dtype() == AITER_DTYPE_i64 && positions->dim() == 1 &&
                        positions->size(0) >= N && positions->is_contiguous(),
                    "inv_rope_positions must be a contiguous int64 [>= N] tensor");
        // read as 16B vectors
        AITER_CHECK(freqs->dtype() == AITER_DTYPE_fp32 && freqs->dim() == 2 &&
                        freqs->size(1) == 64 && freqs->stride(1) == 1 &&
                        freqs->stride(0) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(freqs->data_ptr()) % 16 == 0,
                    "inv_rope_freqs must be fp32 [max_pos, 64] (view_as_real(freqs_cis)), "
                    "16B-aligned rows with the last dim contiguous");
        epi.positions         = reinterpret_cast<const int64_t*>(positions->data_ptr());
        epi.rope_freqs        = reinterpret_cast<const float*>(freqs->data_ptr());
        epi.stride_rope_freqs = static_cast<int>(freqs->stride(0));
    }
    if(out_scale)
    {
        // a head row's 4 scales may go out as one dword
        AITER_CHECK(out_scale->dtype() == AITER_DTYPE_u8 && out_scale->is_contiguous() &&
                        static_cast<int64_t>(out_scale->numel()) ==
                            static_cast<int64_t>(N) * H * (D / 128) &&
                        reinterpret_cast<uintptr_t>(out_scale->data_ptr()) % 4 == 0,
                    "out_scale must be a contiguous, 4B-aligned uint8 tensor of "
                    "N * H * D / 128 e8m0 scales");
        epi.out_scale = reinterpret_cast<uint8_t*>(out_scale->data_ptr());
    }
}

// Calls f(std::bool_constant<INV_ROPE>{}, std::bool_constant<OUT_MXFP8>{}).
template <class F>
void dispatch_epilogue(bool inv_rope, bool out_mxfp8, F&& f)
{
    using Off = std::false_type;
    using On  = std::true_type;
    if(!inv_rope && !out_mxfp8)
        f(Off{}, Off{});
    else if(!out_mxfp8)
        f(On{}, Off{});
    else if(inv_rope)
        f(On{}, On{});
    else
        f(Off{}, On{});
}

} // namespace

void opus_mla_v4_prefill_a16w16_gfx950_fwd(aiter_tensor_t& q,
                                           aiter_tensor_t& unified_kv,
                                           aiter_tensor_t& kv_indices_prefix,
                                           aiter_tensor_t& kv_indptr_prefix,
                                           aiter_tensor_t& kv,
                                           aiter_tensor_t& kv_indices_extend,
                                           aiter_tensor_t& kv_indptr_extend,
                                           aiter_tensor_t& attn_sink,
                                           aiter_tensor_t& out,
                                           float softmax_scale,
                                           std::optional<aiter_tensor_t> inv_rope_positions,
                                           std::optional<aiter_tensor_t> inv_rope_freqs,
                                           std::optional<aiter_tensor_t> out_scale)
{
    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q.dim() == 3, "q must be 3-D [N, H, D], got ndim=", q.dim());
    AITER_CHECK(unified_kv.dim() == 2,
                "unified_kv must be 2-D [total_pages, D], got ndim=",
                unified_kv.dim());
    AITER_CHECK(kv.dim() == 2,
                "kv must be 2-D [total_tokens, D], got ndim=",
                kv.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, D], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    const bool inv_rope  = inv_rope_positions.has_value();
    const bool out_mxfp8 = out_scale.has_value();
    AITER_CHECK(q.dtype() == kv.dtype() && q.dtype() == unified_kv.dtype(),
                "q/unified_kv/kv must share dtype");
    AITER_CHECK(out.dtype() == (out_mxfp8 ? AITER_DTYPE_fp8 : q.dtype()),
                out_mxfp8 ? "out must be fp8 when out_scale is given" : "out must share q's dtype");
    AITER_CHECK(!(inv_rope || out_mxfp8) || q.dtype() == AITER_DTYPE_bf16,
                "the output epilogue (inv_rope / out_scale) is bf16-only");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16,
                "Only bf16/fp16 are supported");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    AITER_CHECK(D == 512,
                "Only D=512 is compiled for opus_mla_v4_prefill_a16w16_gfx950_fwd, got D=", D);
    AITER_CHECK(unified_kv.size(1) == D, "unified_kv last dim must equal q last dim (D=512)");
    AITER_CHECK(kv.size(1) == D, "kv last dim must equal q last dim (D=512)");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D,
                "out shape must match q [N, H, D]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1,
                "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1,
                "kv_indptr_extend length must be N+1");

    // Row-major contiguous strides are required for Q/UnifiedKV/KV/O along D.
    AITER_CHECK(q.stride(2) == 1 && unified_kv.stride(1) == 1 && kv.stride(1) == 1 &&
                    out.stride(2) == 1,
                "Q/UnifiedKV/KV/O must be contiguous along the head-dim D");

    // Kernel reads these 1-D buffers via raw pointer arithmetic; stride must be 1.
    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend) and attn_sink must be contiguous");

    const int total_pages  = static_cast<int>(unified_kv.size(0));
    const int total_tokens = static_cast<int>(kv.size(0));

    if (N == 0) return;

    // ---- Build kernel args -----------------------------------------------
    // The epilogue fields are only passed on when an epilogue is requested.
    opus_mla_v4_prefill_epilogue_kargs<opus_mla_v4_prefill_kargs> kargs{};
    kargs.q_ptr             = q.data_ptr();
    kargs.unified_kv_ptr    = unified_kv.data_ptr();
    kargs.kv_ptr            = kv.data_ptr();
    kargs.attn_sink_ptr     = attn_sink.data_ptr();
    kargs.out_ptr           = out.data_ptr();
    kargs.kv_indptr_prefix  = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    kargs.kv_indices_prefix = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    kargs.kv_indptr_extend  = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    kargs.kv_indices_extend = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    kargs.N                 = N;
    kargs.H                 = H;
    kargs.D                 = D;
    kargs.total_pages       = total_pages;
    kargs.total_tokens      = total_tokens;
    // The kernel assumes the standard row-major layout for [N, H, D] with the
    // head dim contiguous; we already enforced stride(D) == 1 above.
    kargs.stride_qo_n       = static_cast<int>(q.stride(0));
    kargs.stride_qo_h       = static_cast<int>(q.stride(1));
    kargs.stride_kv_page    = static_cast<int>(unified_kv.stride(0));
    AITER_CHECK(kargs.stride_kv_page == static_cast<int>(kv.stride(0)),
                "unified_kv and kv must share row stride along the D dim");
    kargs.softmax_scale     = softmax_scale;

    set_epilogue_args(kargs, inv_rope_positions, inv_rope_freqs, out_scale, N, H, D);
    AITER_CHECK(!out_mxfp8 || (out.stride(0) == q.stride(0) && out.stride(1) == q.stride(1)),
                "fp8 out must share q's strides");

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_OPUS_MLA_V4_PREFILL(KERNEL, TRAITS, KV_TILE, NUM_WARPS)                   \
    do {                                                                                 \
        auto launch = [&](auto dtype_tag, auto inv_rope_c, auto out_mxfp8_c) {           \
            using Traits = TRAITS<16, KV_TILE, 512, NUM_WARPS, decltype(dtype_tag)>;     \
            constexpr bool INV_ROPE  = decltype(inv_rope_c)::value;                      \
            constexpr bool OUT_MXFP8 = decltype(out_mxfp8_c)::value;                     \
            using KArgs = opus_mla_v4_prefill_kargs_t<opus_mla_v4_prefill_kargs,         \
                                                      INV_ROPE || OUT_MXFP8>;            \
            const int num_h_blocks = ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M);     \
            dim3 grid(N, num_h_blocks, 1);                                               \
            dim3 block(Traits::BLOCK_SIZE);                                              \
            KERNEL<Traits, INV_ROPE, OUT_MXFP8>                                          \
                <<<grid, block, 0, stream>>>(static_cast<const KArgs&>(kargs));          \
            HIP_CALL_LAUNCH(hipGetLastError());                                          \
        };                                                                               \
        if(q.dtype() == AITER_DTYPE_fp16)                                                \
            launch(fp16_t{}, std::false_type{}, std::false_type{});                      \
        else                                                                             \
            dispatch_epilogue(inv_rope, out_mxfp8, [&](auto r, auto m) {                 \
                launch(bf16_t{}, r, m);                                                  \
            });                                                                          \
    } while(0)

    // 16mx8_32nx1 (T_M=NUM_WARPS) for H > 32; 16mx1_16nx4 (T_M=1) for H <= 32.
    if(H <= 32)
        LAUNCH_OPUS_MLA_V4_PREFILL(opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel,
                                   opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits, 64, 4);
    else
        LAUNCH_OPUS_MLA_V4_PREFILL(opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernel,
                                   opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits, 32, 8);

#undef LAUNCH_OPUS_MLA_V4_PREFILL
}

void opus_mla_v4_prefill_a8w8_gfx950_fwd(aiter_tensor_t& q_nope,
                                         aiter_tensor_t& q_rope,
                                         aiter_tensor_t& unified_kv_nope,
                                         aiter_tensor_t& unified_kv_rope,
                                         aiter_tensor_t& kv_indices_prefix,
                                         aiter_tensor_t& kv_indptr_prefix,
                                         aiter_tensor_t& kv_nope,
                                         aiter_tensor_t& kv_rope,
                                         aiter_tensor_t& kv_indices_extend,
                                         aiter_tensor_t& kv_indptr_extend,
                                         aiter_tensor_t& attn_sink,
                                         aiter_tensor_t& out,
                                         float softmax_scale,
                                         std::optional<aiter_tensor_t> inv_rope_positions,
                                         std::optional<aiter_tensor_t> inv_rope_freqs,
                                         std::optional<aiter_tensor_t> out_scale)
{
    // Single compiled configuration: split NoPE fp8 (448 + 14 E8M0 scales + pad
    // = 512 fp8 slots/row) and RoPE bf16 (64), D_HEAD = 512.
    using Traits = opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>;
    constexpr int D_NOPE_PADDED = Traits::D_NOPE_PADDED_SIZE; // 512
    constexpr int D_ROPE        = Traits::D_ROPE_SIZE;        // 64
    constexpr int D_HEAD        = Traits::D_HEAD_SIZE;        // 512

    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q_nope.dim() == 3, "q_nope must be 3-D [N, H, 512], got ndim=", q_nope.dim());
    AITER_CHECK(q_rope.dim() == 3, "q_rope must be 3-D [N, H, 64], got ndim=", q_rope.dim());
    AITER_CHECK(unified_kv_nope.dim() == 2,
                "unified_kv_nope must be 2-D [total_pages, 512], got ndim=", unified_kv_nope.dim());
    AITER_CHECK(unified_kv_rope.dim() == 2,
                "unified_kv_rope must be 2-D [total_pages, 64], got ndim=", unified_kv_rope.dim());
    AITER_CHECK(kv_nope.dim() == 2,
                "kv_nope must be 2-D [total_tokens, 512], got ndim=", kv_nope.dim());
    AITER_CHECK(kv_rope.dim() == 2,
                "kv_rope must be 2-D [total_tokens, 64], got ndim=", kv_rope.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, 512], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    AITER_CHECK(q_nope.dtype() == AITER_DTYPE_fp8 && unified_kv_nope.dtype() == AITER_DTYPE_fp8 &&
                    kv_nope.dtype() == AITER_DTYPE_fp8,
                "q_nope/unified_kv_nope/kv_nope must be fp8");
    AITER_CHECK(q_rope.dtype() == AITER_DTYPE_bf16 && unified_kv_rope.dtype() == AITER_DTYPE_bf16 &&
                    kv_rope.dtype() == AITER_DTYPE_bf16,
                "q_rope/unified_kv_rope/kv_rope must be bf16");
    const bool inv_rope  = inv_rope_positions.has_value();
    const bool out_mxfp8 = out_scale.has_value();
    AITER_CHECK(out.dtype() == (out_mxfp8 ? AITER_DTYPE_fp8 : AITER_DTYPE_bf16),
                out_mxfp8 ? "out must be fp8 when out_scale is given" : "out must be bf16");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q_nope.size(0));
    const int H = static_cast<int>(q_nope.size(1));

    AITER_CHECK(q_nope.size(2) == D_NOPE_PADDED, "q_nope last dim must be 512 (NoPE padded + scales)");
    AITER_CHECK(q_rope.size(0) == N && q_rope.size(1) == H && q_rope.size(2) == D_ROPE,
                "q_rope shape must be [N, H, 64]");
    AITER_CHECK(unified_kv_nope.size(1) == D_NOPE_PADDED, "unified_kv_nope last dim must be 512");
    AITER_CHECK(unified_kv_rope.size(1) == D_ROPE, "unified_kv_rope last dim must be 64");
    AITER_CHECK(kv_nope.size(1) == D_NOPE_PADDED, "kv_nope last dim must be 512");
    AITER_CHECK(kv_rope.size(1) == D_ROPE, "kv_rope last dim must be 64");
    AITER_CHECK(unified_kv_nope.size(0) == unified_kv_rope.size(0),
                "unified_kv_nope and unified_kv_rope must share total_pages");
    AITER_CHECK(kv_nope.size(0) == kv_rope.size(0),
                "kv_nope and kv_rope must share total_tokens");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D_HEAD,
                "out shape must be [N, H, 512]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1, "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1, "kv_indptr_extend length must be N+1");

    // The kernel indexes consecutive query heads within a tile by D_NOPE_PADDED /
    // D_ROPE; Q/KV NoPE/RoPE rows must therefore be densely packed.
    AITER_CHECK(q_nope.stride(2) == 1 && q_nope.stride(1) == D_NOPE_PADDED,
                "q_nope must be contiguous with row stride 512");
    AITER_CHECK(q_rope.stride(2) == 1 && q_rope.stride(1) == D_ROPE,
                "q_rope must be contiguous with row stride 64");
    AITER_CHECK(unified_kv_nope.stride(1) == 1 && kv_nope.stride(1) == 1,
                "kv_nope/unified_kv_nope must be contiguous along the head-dim");
    AITER_CHECK(unified_kv_rope.stride(1) == 1 && kv_rope.stride(1) == 1,
                "kv_rope/unified_kv_rope must be contiguous along the head-dim");
    AITER_CHECK(out.stride(2) == 1, "out must be contiguous along the head-dim");

    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend) and attn_sink must be contiguous");

    const int total_pages  = static_cast<int>(unified_kv_nope.size(0));
    const int total_tokens = static_cast<int>(kv_nope.size(0));

    if(N == 0)
        return;

    const int stride_kv_nope_page = static_cast<int>(unified_kv_nope.stride(0));
    const int stride_kv_rope_page = static_cast<int>(unified_kv_rope.stride(0));
    AITER_CHECK(stride_kv_nope_page == static_cast<int>(kv_nope.stride(0)),
                "unified_kv_nope and kv_nope must share row stride");
    AITER_CHECK(stride_kv_rope_page == static_cast<int>(kv_rope.stride(0)),
                "unified_kv_rope and kv_rope must share row stride");

    // ---- Build kernel args -----------------------------------------------
    // The epilogue fields are only passed on when an epilogue is requested.
    opus_mla_v4_prefill_epilogue_kargs<opus_mla_v4_prefill_fp8_kargs> kargs{};
    kargs.q_nope_ptr          = q_nope.data_ptr();
    kargs.q_rope_ptr          = q_rope.data_ptr();
    kargs.unified_kv_nope_ptr = unified_kv_nope.data_ptr();
    kargs.unified_kv_rope_ptr = unified_kv_rope.data_ptr();
    kargs.kv_nope_ptr         = kv_nope.data_ptr();
    kargs.kv_rope_ptr         = kv_rope.data_ptr();
    kargs.attn_sink_ptr       = attn_sink.data_ptr();
    kargs.out_ptr             = out.data_ptr();
    kargs.kv_indptr_prefix    = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    kargs.kv_indices_prefix   = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    kargs.kv_indptr_extend    = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    kargs.kv_indices_extend   = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    kargs.N                   = N;
    kargs.H                   = H;
    kargs.total_pages         = total_pages;
    kargs.total_tokens        = total_tokens;
    kargs.stride_q_nope_n     = static_cast<int>(q_nope.stride(0));
    kargs.stride_q_nope_h     = static_cast<int>(q_nope.stride(1));
    kargs.stride_q_rope_n     = static_cast<int>(q_rope.stride(0));
    kargs.stride_q_rope_h     = static_cast<int>(q_rope.stride(1));
    kargs.stride_o_n          = static_cast<int>(out.stride(0));
    kargs.stride_o_h          = static_cast<int>(out.stride(1));
    kargs.stride_kv_nope_page = stride_kv_nope_page;
    kargs.stride_kv_rope_page = stride_kv_rope_page;
    kargs.softmax_scale       = softmax_scale;
    set_epilogue_args(kargs, inv_rope_positions, inv_rope_freqs, out_scale, N, H, D_HEAD);

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q_nope.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_OPUS_MLA_V4_PREFILL_FP8(KERNEL, TRAITS, KV_TILE, NUM_WARPS)                 \
    dispatch_epilogue(inv_rope, out_mxfp8, [&](auto inv_rope_c, auto out_mxfp8_c) {       \
        using KTraits = TRAITS<16, KV_TILE, NUM_WARPS, fp8_t, bf16_t, bf16_t>;            \
        constexpr bool INV_ROPE  = decltype(inv_rope_c)::value;                           \
        constexpr bool OUT_MXFP8 = decltype(out_mxfp8_c)::value;                          \
        using KArgs = opus_mla_v4_prefill_kargs_t<opus_mla_v4_prefill_fp8_kargs,          \
                                                  INV_ROPE || OUT_MXFP8>;                 \
        const int num_h_blocks = ceil_div(H, KTraits::Q_TILE_SIZE * KTraits::T_M);        \
        dim3 grid(N, num_h_blocks, 1);                                                    \
        dim3 block(KTraits::BLOCK_SIZE);                                                  \
        KERNEL<KTraits, INV_ROPE, OUT_MXFP8>                                              \
            <<<grid, block, 0, stream>>>(static_cast<const KArgs&>(kargs));               \
        HIP_CALL_LAUNCH(hipGetLastError());                                               \
    })

    // 16mx8_32nx1 (T_M=NUM_WARPS) for H > 32; 16mx1_16nx4 (T_M=1) for H <= 32.
    if(H <= 32)
        LAUNCH_OPUS_MLA_V4_PREFILL_FP8(opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel,
                                       opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits, 64, 4);
    else
        LAUNCH_OPUS_MLA_V4_PREFILL_FP8(opus_mla_v4_prefill_a8w8_16mx8_32nx1_kernel,
                                       opus_mla_v4_prefill_a8w8_16mx8_32nx1_traits, 32, 8);

#undef LAUNCH_OPUS_MLA_V4_PREFILL_FP8
}

// ============================================================================
// gfx1250: prebuilt code object
// ============================================================================

// The code objects' kernel arguments are a field-for-field match of
// opus_mla_v4_prefill_kargs and opus_mla_v4_prefill_fp8_kargs above, so those are
// reused verbatim as the kernarg buffers.
//
// The catch is that nothing in this repo rebuilds the code objects: an edit made
// for the gfx950 path would silently corrupt the gfx1250 launch. Pin every field
// so such an edit fails the build instead. Sizes match the
// `.kernarg_segment_size` reported by `llvm-readelf --notes <code object>`.
#define OPUS_MLA_V4_GFX1250_CO_ABI(struct_, field_, offset_)                      \
    static_assert(offsetof(struct_, field_) == (offset_),                         \
                  #struct_ "::" #field_ " moved; rebuild the gfx1250 code objects")

static_assert(sizeof(opus_mla_v4_prefill_kargs) == 112,
              "opus_mla_v4_prefill_kargs resized; rebuild the gfx1250 code objects");
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, q_ptr, 0);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, unified_kv_ptr, 8);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, kv_ptr, 16);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, attn_sink_ptr, 24);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, out_ptr, 32);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, kv_indptr_prefix, 40);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, kv_indices_prefix, 48);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, kv_indptr_extend, 56);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, kv_indices_extend, 64);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, N, 72);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, H, 76);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, D, 80);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, total_pages, 84);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, total_tokens, 88);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, stride_qo_n, 92);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, stride_qo_h, 96);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, stride_kv_page, 100);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_kargs, softmax_scale, 104);

static_assert(sizeof(opus_mla_v4_prefill_fp8_kargs) == 152,
              "opus_mla_v4_prefill_fp8_kargs resized; rebuild the gfx1250 code objects");
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, q_nope_ptr, 0);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, q_rope_ptr, 8);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, unified_kv_nope_ptr, 16);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, unified_kv_rope_ptr, 24);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, kv_nope_ptr, 32);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, kv_rope_ptr, 40);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, attn_sink_ptr, 48);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, out_ptr, 56);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, kv_indptr_prefix, 64);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, kv_indices_prefix, 72);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, kv_indptr_extend, 80);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, kv_indices_extend, 88);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, N, 96);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, H, 100);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, total_pages, 104);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, total_tokens, 108);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_q_nope_n, 112);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_q_nope_h, 116);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_q_rope_n, 120);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_q_rope_h, 124);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_o_n, 128);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_o_h, 132);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_kv_nope_page, 136);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, stride_kv_rope_page, 140);
OPUS_MLA_V4_GFX1250_CO_ABI(opus_mla_v4_prefill_fp8_kargs, softmax_scale, 144);

#undef OPUS_MLA_V4_GFX1250_CO_ABI

namespace {

// Launch geometry of the 16mx4_64nx1 code objects: one workgroup covers one
// query token and Q_TILE_SIZE * T_M query heads.
constexpr int kGfx1250QTileSize   = 16;
constexpr int kGfx1250NumWarps    = 4;
constexpr int kGfx1250WarpSize    = 32; // wave32 on gfx1250
constexpr int kGfx1250BlockSize   = kGfx1250NumWarps * kGfx1250WarpSize;
constexpr int kGfx1250HeadsPerBlk = kGfx1250QTileSize * kGfx1250NumWarps;
constexpr int kGfx1250MaxClusterY = 2;

// Narrow-head variants: T_M=1, so heads per workgroup is Q_TILE_SIZE alone.
constexpr int kGfx1250Heads16mx1 = 16;
constexpr int kGfx1250Heads32mx1 = 32;

// fp8 split-precision head layout.
constexpr int kGfx1250DNopePadded = 512;
constexpr int kGfx1250DRope       = 64;
constexpr int kGfx1250DHead       = 512;

// Each symbol name is its code object's file stem plus an optional `_cyN` and `_kernel`.
constexpr const char* kGfx1250A16W16Co =
    "mla_v4_opus/opus_mla_v4_prefill_a16w16_16mx4_64nx1.co";
constexpr const char* kGfx1250A16W16Co16mx1 =
    "mla_v4_opus/opus_mla_v4_prefill_a16w16_16mx1_16nx4.co";
constexpr const char* kGfx1250A16W16Co32mx1 =
    "mla_v4_opus/opus_mla_v4_prefill_a16w16_32mx1_16nx4.co";
constexpr const char* kGfx1250A8W8Co =
    "mla_v4_opus/opus_mla_v4_prefill_a8w8_16mx4_64nx1.co";
constexpr const char* kGfx1250A8W8Co16mx1 =
    "mla_v4_opus/opus_mla_v4_prefill_a8w8_16mx1_16nx4.co";
constexpr const char* kGfx1250A8W8Co32mx1 =
    "mla_v4_opus/opus_mla_v4_prefill_a8w8_32mx1_16nx4.co";

// OPUS-managed prebuilt code objects: reuse AiterAsmKernel's file loader but opt
// OUT of the gfx1250 B0-only asm gate (arch coverage is the OPUS layer's job).
struct PrebuiltKernel : AiterAsmKernel
{
    PrebuiltKernel(const char* kernel_name, const char* hsaco_path)
        : AiterAsmKernel(kernel_name, hsaco_path, AiterAsmKernel::SkipGfx1250Gate{})
    {
    }
};

// Largest cluster width that evenly divides grid.y. A cluster that does not
// divide grid.y would gather through a masked peer set and silently corrupt the
// tail workgroup, so the kernel only ever sees an exact divisor.
int gfx1250_pick_cluster_y(int num_h_blocks)
{
    for(int c = kGfx1250MaxClusterY; c > 1; c >>= 1)
        if(num_h_blocks % c == 0)
            return c;
    return 1;
}

} // namespace

void opus_mla_v4_prefill_a16w16_gfx1250_fwd(aiter_tensor_t& q,
                                            aiter_tensor_t& unified_kv,
                                            aiter_tensor_t& kv_indices_prefix,
                                            aiter_tensor_t& kv_indptr_prefix,
                                            aiter_tensor_t& kv,
                                            aiter_tensor_t& kv_indices_extend,
                                            aiter_tensor_t& kv_indptr_extend,
                                            aiter_tensor_t& attn_sink,
                                            aiter_tensor_t& out,
                                            float softmax_scale)
{
    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q.dim() == 3, "q must be 3-D [N, H, D], got ndim=", q.dim());
    AITER_CHECK(unified_kv.dim() == 2,
                "unified_kv must be 2-D [total_pages, D], got ndim=",
                unified_kv.dim());
    AITER_CHECK(kv.dim() == 2, "kv must be 2-D [total_tokens, D], got ndim=", kv.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, D], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    AITER_CHECK(q.dtype() == kv.dtype() && q.dtype() == unified_kv.dtype() &&
                    q.dtype() == out.dtype(),
                "q/unified_kv/kv/out must share dtype");
    // Only the bf16 traits are instantiated in the gfx1250 code object.
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16,
                "the gfx1250 code object only provides the bf16 variant");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    AITER_CHECK(D == 512, "Only D=512 is built into the gfx1250 code object, got D=", D);
    AITER_CHECK(unified_kv.size(1) == D, "unified_kv last dim must equal q last dim (D=512)");
    AITER_CHECK(kv.size(1) == D, "kv last dim must equal q last dim (D=512)");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D,
                "out shape must match q [N, H, D]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1, "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1, "kv_indptr_extend length must be N+1");

    AITER_CHECK(q.stride(2) == 1 && unified_kv.stride(1) == 1 && kv.stride(1) == 1 &&
                    out.stride(2) == 1,
                "Q/UnifiedKV/KV/O must be contiguous along the head-dim D");
    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend) and attn_sink must be contiguous");

    if(N == 0)
        return;

    // ---- Build kernel args -----------------------------------------------
    opus_mla_v4_prefill_kargs args{};
    args.q_ptr             = q.data_ptr();
    args.unified_kv_ptr    = unified_kv.data_ptr();
    args.kv_ptr            = kv.data_ptr();
    args.attn_sink_ptr     = attn_sink.data_ptr();
    args.out_ptr           = out.data_ptr();
    args.kv_indptr_prefix  = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    args.kv_indices_prefix = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    args.kv_indptr_extend  = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    args.kv_indices_extend = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    args.N                 = N;
    args.H                 = H;
    args.D                 = D;
    args.total_pages       = static_cast<int>(unified_kv.size(0));
    args.total_tokens      = static_cast<int>(kv.size(0));
    args.stride_qo_n       = static_cast<int>(q.stride(0));
    args.stride_qo_h       = static_cast<int>(q.stride(1));
    args.stride_kv_page    = static_cast<int>(unified_kv.stride(0));
    AITER_CHECK(args.stride_kv_page == static_cast<int>(kv.stride(0)),
                "unified_kv and kv must share row stride along the D dim");
    args.softmax_scale = softmax_scale;

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    size_t arg_size = sizeof(args);

    // 16mx1_16nx4 for H <= 16, 32mx1_16nx4 for H <= 32, else the clustered 16mx4_64nx1.
    if(H <= kGfx1250Heads16mx1)
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel",
                                   kGfx1250A16W16Co16mx1);
        impl.launch_kernel({&args,
                            &arg_size,
                            N,
                            ceil_div(H, kGfx1250Heads16mx1),
                            1,
                            kGfx1250BlockSize,
                            1,
                            1,
                            stream});
        return;
    }

    if(H <= kGfx1250Heads32mx1)
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a16w16_32mx1_16nx4_kernel",
                                   kGfx1250A16W16Co32mx1);
        impl.launch_kernel({&args,
                            &arg_size,
                            N,
                            ceil_div(H, kGfx1250Heads32mx1),
                            1,
                            kGfx1250BlockSize,
                            1,
                            1,
                            stream});
        return;
    }

    const int num_h_blocks = ceil_div(H, kGfx1250HeadsPerBlk);
    const int cluster_y    = gfx1250_pick_cluster_y(num_h_blocks);

    if(cluster_y == 2)
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a16w16_16mx4_64nx1_cy2_kernel",
                                   kGfx1250A16W16Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream, 1, 2, 1});
    }
    else
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a16w16_16mx4_64nx1_cy1_kernel",
                                   kGfx1250A16W16Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream});
    }
}

void opus_mla_v4_prefill_a8w8_gfx1250_fwd(aiter_tensor_t& q_nope,
                                          aiter_tensor_t& q_rope,
                                          aiter_tensor_t& unified_kv_nope,
                                          aiter_tensor_t& unified_kv_rope,
                                          aiter_tensor_t& kv_indices_prefix,
                                          aiter_tensor_t& kv_indptr_prefix,
                                          aiter_tensor_t& kv_nope,
                                          aiter_tensor_t& kv_rope,
                                          aiter_tensor_t& kv_indices_extend,
                                          aiter_tensor_t& kv_indptr_extend,
                                          aiter_tensor_t& attn_sink,
                                          aiter_tensor_t& out,
                                          float softmax_scale)
{
    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q_nope.dim() == 3, "q_nope must be 3-D [N, H, 512], got ndim=", q_nope.dim());
    AITER_CHECK(q_rope.dim() == 3, "q_rope must be 3-D [N, H, 64], got ndim=", q_rope.dim());
    AITER_CHECK(unified_kv_nope.dim() == 2,
                "unified_kv_nope must be 2-D [total_pages, 512], got ndim=",
                unified_kv_nope.dim());
    AITER_CHECK(unified_kv_rope.dim() == 2,
                "unified_kv_rope must be 2-D [total_pages, 64], got ndim=",
                unified_kv_rope.dim());
    AITER_CHECK(kv_nope.dim() == 2,
                "kv_nope must be 2-D [total_tokens, 512], got ndim=",
                kv_nope.dim());
    AITER_CHECK(kv_rope.dim() == 2,
                "kv_rope must be 2-D [total_tokens, 64], got ndim=",
                kv_rope.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, 512], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    AITER_CHECK(q_nope.dtype() == AITER_DTYPE_fp8 && unified_kv_nope.dtype() == AITER_DTYPE_fp8 &&
                    kv_nope.dtype() == AITER_DTYPE_fp8,
                "q_nope/unified_kv_nope/kv_nope must be fp8");
    AITER_CHECK(q_rope.dtype() == AITER_DTYPE_bf16 && unified_kv_rope.dtype() == AITER_DTYPE_bf16 &&
                    kv_rope.dtype() == AITER_DTYPE_bf16,
                "q_rope/unified_kv_rope/kv_rope must be bf16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q_nope.size(0));
    const int H = static_cast<int>(q_nope.size(1));

    AITER_CHECK(q_nope.size(2) == kGfx1250DNopePadded,
                "q_nope last dim must be 512 (NoPE padded + scales)");
    AITER_CHECK(q_rope.size(0) == N && q_rope.size(1) == H && q_rope.size(2) == kGfx1250DRope,
                "q_rope shape must be [N, H, 64]");
    AITER_CHECK(unified_kv_nope.size(1) == kGfx1250DNopePadded,
                "unified_kv_nope last dim must be 512");
    AITER_CHECK(unified_kv_rope.size(1) == kGfx1250DRope, "unified_kv_rope last dim must be 64");
    AITER_CHECK(kv_nope.size(1) == kGfx1250DNopePadded, "kv_nope last dim must be 512");
    AITER_CHECK(kv_rope.size(1) == kGfx1250DRope, "kv_rope last dim must be 64");
    AITER_CHECK(unified_kv_nope.size(0) == unified_kv_rope.size(0),
                "unified_kv_nope and unified_kv_rope must share total_pages");
    AITER_CHECK(kv_nope.size(0) == kv_rope.size(0),
                "kv_nope and kv_rope must share total_tokens");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == kGfx1250DHead,
                "out shape must be [N, H, 512]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1, "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1, "kv_indptr_extend length must be N+1");

    AITER_CHECK(q_nope.stride(2) == 1 && q_nope.stride(1) == kGfx1250DNopePadded,
                "q_nope must be contiguous with row stride 512");
    AITER_CHECK(q_rope.stride(2) == 1 && q_rope.stride(1) == kGfx1250DRope,
                "q_rope must be contiguous with row stride 64");
    AITER_CHECK(unified_kv_nope.stride(1) == 1 && kv_nope.stride(1) == 1,
                "kv_nope/unified_kv_nope must be contiguous along the head-dim");
    AITER_CHECK(unified_kv_rope.stride(1) == 1 && kv_rope.stride(1) == 1,
                "kv_rope/unified_kv_rope must be contiguous along the head-dim");
    AITER_CHECK(out.stride(2) == 1, "out must be contiguous along the head-dim");

    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend) and attn_sink must be contiguous");

    if(N == 0)
        return;

    const int stride_kv_nope_page = static_cast<int>(unified_kv_nope.stride(0));
    const int stride_kv_rope_page = static_cast<int>(unified_kv_rope.stride(0));
    AITER_CHECK(stride_kv_nope_page == static_cast<int>(kv_nope.stride(0)),
                "unified_kv_nope and kv_nope must share row stride");
    AITER_CHECK(stride_kv_rope_page == static_cast<int>(kv_rope.stride(0)),
                "unified_kv_rope and kv_rope must share row stride");

    // ---- Build kernel args -----------------------------------------------
    opus_mla_v4_prefill_fp8_kargs args{};
    args.q_nope_ptr          = q_nope.data_ptr();
    args.q_rope_ptr          = q_rope.data_ptr();
    args.unified_kv_nope_ptr = unified_kv_nope.data_ptr();
    args.unified_kv_rope_ptr = unified_kv_rope.data_ptr();
    args.kv_nope_ptr         = kv_nope.data_ptr();
    args.kv_rope_ptr         = kv_rope.data_ptr();
    args.attn_sink_ptr       = attn_sink.data_ptr();
    args.out_ptr             = out.data_ptr();
    args.kv_indptr_prefix    = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    args.kv_indices_prefix   = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    args.kv_indptr_extend    = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    args.kv_indices_extend   = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    args.N                   = N;
    args.H                   = H;
    args.total_pages         = static_cast<int>(unified_kv_nope.size(0));
    args.total_tokens        = static_cast<int>(kv_nope.size(0));
    args.stride_q_nope_n     = static_cast<int>(q_nope.stride(0));
    args.stride_q_nope_h     = static_cast<int>(q_nope.stride(1));
    args.stride_q_rope_n     = static_cast<int>(q_rope.stride(0));
    args.stride_q_rope_h     = static_cast<int>(q_rope.stride(1));
    args.stride_o_n          = static_cast<int>(out.stride(0));
    args.stride_o_h          = static_cast<int>(out.stride(1));
    args.stride_kv_nope_page = stride_kv_nope_page;
    args.stride_kv_rope_page = stride_kv_rope_page;
    args.softmax_scale       = softmax_scale;

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q_nope.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    size_t arg_size = sizeof(args);

    // 16mx1_16nx4 for H <= 16, 32mx1_16nx4 for H <= 32, else the clustered 16mx4_64nx1.
    if(H <= kGfx1250Heads16mx1)
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel",
                                   kGfx1250A8W8Co16mx1);
        impl.launch_kernel({&args,
                            &arg_size,
                            N,
                            ceil_div(H, kGfx1250Heads16mx1),
                            1,
                            kGfx1250BlockSize,
                            1,
                            1,
                            stream});
        return;
    }

    if(H <= kGfx1250Heads32mx1)
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a8w8_32mx1_16nx4_kernel",
                                   kGfx1250A8W8Co32mx1);
        impl.launch_kernel({&args,
                            &arg_size,
                            N,
                            ceil_div(H, kGfx1250Heads32mx1),
                            1,
                            kGfx1250BlockSize,
                            1,
                            1,
                            stream});
        return;
    }

    const int num_h_blocks = ceil_div(H, kGfx1250HeadsPerBlk);
    const int cluster_y    = gfx1250_pick_cluster_y(num_h_blocks);

    if(cluster_y == 2)
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_kernel",
                                   kGfx1250A8W8Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream, 1, 2, 1});
    }
    else
    {
        static PrebuiltKernel impl("opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_kernel",
                                   kGfx1250A8W8Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream});
    }
}
