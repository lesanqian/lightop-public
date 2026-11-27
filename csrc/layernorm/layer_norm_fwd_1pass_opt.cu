// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/cuda.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM) || defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <optional>
#include <string>
#include <type_traits>

#if defined(TORCH_VERSION_OVER_2120)
// torch>=2.12 removed C10_WARP_SIZE_STATIC, wavefront size is fixed at 64
#define C10_WARP_SIZE 64
#elif defined(TORCH_VERSION_OVER_290)
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {

enum class GateActivation : int {
  kNone = 0,
  kSilu = 1,
  kSigmoid = 2,
};

template <typename scalar_t>
static __device__ __forceinline__ float ln1p_to_float(const scalar_t in) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    return static_cast<float>(in);
  } else {
    union {
      uint32_t int32;
      float fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}

template <typename scalar_t, typename value_t>
static __device__ __forceinline__ void ln1p_from_float(scalar_t& out,
                                                       const value_t value) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    out = static_cast<scalar_t>(value);
  } else {
    const float f = static_cast<float>(value);
    uint32_t u = *(uint32_t*)(&f);
    u += 0x8000;
    out = u >> 16;
  }
}

namespace ln1p_gfx946 {
#if defined(__gfx946__)
// ============================================================================
// gfx946 (Shaobo) Gated RMSNorm / LayerNorm hardware primitives
//   Design constraints (ISA v1.8 & DTK 27.04-1010/1114 verified errata):
//   1. 512 KB VGPR/CU (512 VGPR/SIMD, alloc gran 4): keep raw packed BF16/FP16
//      in uint32_t[Pairs] across reduction when IS_RMS_NORM && (NORM_BEFORE_GATE
//      || !HAS_Z) so peak VGPR stays <= 64 (8 waves/SIMD, SCRATCH=0).
//   2. Pass-1 sum-of-squares uses VOP3P v_dot2_f32_bf16 / __builtin_hcu_fdot2
//      ONLY when input is raw BF16/FP16 (NEVER when !NORM_BEFORE_GATE && HAS_Z,
//      which must keep FP32 x * gate(z) precision).
//   3. Packed FP32 math uses ext_vector_type(2) (v2f) to auto-emit VOP3P
//      v_pk_{add,mul,fma}_f32 under -O3 (avoids missing .def explicit builtins).
//   4. Packed output writeback uses single-insn VOP3A RNE v_cvt_pk_{bf16,f16}_f32
//      (never v_cvt_pkrtz_* RTZ).
//   5. Cross-half-wave reductions use dual "+v" asm v_permlane{16,32}_swap_b32
//      (errata R4: avoids broken frontend builtin missing 2nd writeback) paired
//      with intra-row DPP (0xB1, 0x4E, 0x141, 0x140) -> zero ds_bpermute_b32.
//   6. Sigmoid reciprocal uses 1-insn v_rcp_f32 (<1 ULP; denom = 1+exp2 >= 1)
//      replacing ~10-insn IEEE div + VCC 4-NOP hazard; batched VALU ahead of
//      quad-rate SPF clears the §4.2.3 VALU->SPF 1-NOP hazard.
// ============================================================================
using v2f = float __attribute__((ext_vector_type(2)));
using v2h = _Float16 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ v2f bf16x2_to_v2f(uint32_t packed) {
  const uint32_t lo = packed << 16;
  const uint32_t hi = packed & 0xFFFF0000U;
  return v2f{__uint_as_float(lo), __uint_as_float(hi)};
}

static __device__ __forceinline__ v2f f16x2_to_v2f(uint32_t packed) {
  const __half2 h2 = __builtin_bit_cast(__half2, packed);
  const float2 f = __half22float2(h2);
  return v2f{f.x, f.y};
}

template <bool IS_BF16>
static __device__ __forceinline__ v2f unpack_pair(uint32_t packed) {
  if constexpr (IS_BF16) {
    return bf16x2_to_v2f(packed);
  } else {
    return f16x2_to_v2f(packed);
  }
}

static __device__ __forceinline__ uint32_t v2f_to_bf16x2_rn(v2f v) {
  uint32_t res;
  asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(res) : "v"(v[0]), "v"(v[1]));
  return res;
}

static __device__ __forceinline__ uint32_t v2f_to_f16x2_rn(v2f v) {
  uint32_t res;
  asm("v_cvt_pk_f16_f32 %0, %1, %2" : "=v"(res) : "v"(v[0]), "v"(v[1]));
  return res;
}

template <bool IS_BF16>
static __device__ __forceinline__ uint32_t pack_pair_rn(v2f v) {
  if constexpr (IS_BF16) {
    return v2f_to_bf16x2_rn(v);
  } else {
    return v2f_to_f16x2_rn(v);
  }
}

static __device__ __forceinline__ float dot2_sq_bf16(uint32_t packed_bf16,
                                                     float acc) {
  asm("v_dot2_f32_bf16 %0, %1, %1, %0" : "+v"(acc) : "v"(packed_bf16));
  return acc;
}

static __device__ __forceinline__ float dot2_sq_f16(uint32_t packed_f16,
                                                    float acc) {
#if defined(__has_builtin) && __has_builtin(__builtin_hcu_fdot2)
  const v2h h = __builtin_bit_cast(v2h, packed_f16);
  return __builtin_hcu_fdot2(h, h, acc, false);
#else
  asm("v_dot2_f32_f16 %0, %1, %1, %0" : "+v"(acc) : "v"(packed_f16));
  return acc;
#endif
}

template <bool IS_BF16>
static __device__ __forceinline__ float dot2_sq_pair(uint32_t packed,
                                                     float acc) {
  if constexpr (IS_BF16) {
    return dot2_sq_bf16(packed, acc);
  } else {
    return dot2_sq_f16(packed, acc);
  }
}

template <bool COND>
static __device__ __forceinline__ const uint32_t* opt_ptr(const uint32_t* p,
                                                          const int offset = 0) {
  if constexpr (COND) {
    return p + offset;
  } else {
    return nullptr;
  }
}

static __device__ __forceinline__ float fast_rcpf(float x) {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_rcpf)
  return __builtin_amdgcn_rcpf(x);
#else
  float res;
  asm("v_rcp_f32 %0, %1" : "=v"(res) : "v"(x));
  return res;
#endif
}

template <int kDppCtrl>
static __device__ __forceinline__ float mov_dpp_f32(float val) {
  const int bits = __builtin_bit_cast(int, val);
  const int swapped = __builtin_amdgcn_mov_dpp(bits, kDppCtrl, 0xf, 0xf, true);
  return __builtin_bit_cast(float, swapped);
}

static __device__ __forceinline__ float permlane16_swap_fadd(float val) {
  float other = val;
  asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return val + other;
}

static __device__ __forceinline__ float permlane32_swap_fadd(float val) {
  float other = val;
  asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return val + other;
}

// Pure-VALU all-reduce sum over WIDTH in {2, 4, 8, 16, 32, 64} active lanes:
//   - width >= 2 : quad_perm:[1,0,3,2] (0xB1)
//   - width >= 4 : quad_perm:[2,3,0,1] (0x4E)
//   - width >= 8 : row_half_mirror     (0x141)
//   - width >= 16: row_mirror          (0x140)
//   - width >= 32: v_permlane16_swap_b32
//   - width >= 64: v_permlane32_swap_b32
// Zero ds_bpermute_b32 and zero lgkmcnt stalls. Caller must keep EXEC uniform.
template <int WIDTH>
static __device__ __forceinline__ float valu_allreduce_sum_f32(float val) {
  static_assert(WIDTH == 1 || WIDTH == 2 || WIDTH == 4 || WIDTH == 8 ||
                WIDTH == 16 || WIDTH == 32 || WIDTH == 64);
  if constexpr (WIDTH >= 2) {
    val += mov_dpp_f32<0xb1>(val);
  }
  if constexpr (WIDTH >= 4) {
    val += mov_dpp_f32<0x4e>(val);
  }
  if constexpr (WIDTH >= 8) {
    val += mov_dpp_f32<0x141>(val);
  }
  if constexpr (WIDTH >= 16) {
    val += mov_dpp_f32<0x140>(val);
  }
  if constexpr (WIDTH >= 32) {
    val = permlane16_swap_fadd(val);
  }
  if constexpr (WIDTH >= 64) {
    val = permlane32_swap_fadd(val);
  }
  return val;
}

// Batched pre-gate (Mamba2 !NORM_BEFORE_GATE): vals[p] *= gate(z_raw[p]) in FP32.
template <bool IS_BF16, GateActivation ACT, int N_PAIRS>
static __device__ __forceinline__ void apply_gate_v2f(
    v2f (&vals)[N_PAIRS],
    const uint32_t* z_raw) {
  if constexpr (ACT == GateActivation::kNone) {
    return;
  }
  constexpr float NEG_LOG2E = -1.44269504088896340736f;
  const v2f v_neg_log2e = {NEG_LOG2E, NEG_LOG2E};
  const v2f v_one = {1.0f, 1.0f};
  v2f z_f[N_PAIRS];
  v2f e_f[N_PAIRS];
#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    z_f[p] = unpack_pair<IS_BF16>(z_raw[p]);
    e_f[p] = z_f[p] * v_neg_log2e;
  }
#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    e_f[p] = v2f{__builtin_amdgcn_exp2f(e_f[p][0]),
                 __builtin_amdgcn_exp2f(e_f[p][1])};
  }
#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    e_f[p] = v_one + e_f[p];
  }
#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    e_f[p] = v2f{fast_rcpf(e_f[p][0]), fast_rcpf(e_f[p][1])};
  }
#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    if constexpr (ACT == GateActivation::kSilu) {
      const v2f gate = z_f[p] * e_f[p];
      vals[p] = vals[p] * gate;
    } else {
      vals[p] = vals[p] * e_f[p];
    }
  }
}

// Fused Pass-2 epilogue from raw packed BF16/FP16 x_raw (IS_RMS_NORM && (NORM_BEFORE_GATE || !HAS_Z)).
// Interleaves (x*rstd)*w(+b) VALU with v_exp_f32 and batches VALU ahead of SPF.
template <bool IS_BF16, bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT, int N_PAIRS>
static __device__ __forceinline__ void norm_epilogue_from_raw(
    const uint32_t* x_raw,
    const uint32_t* w_raw,
    const uint32_t* b_raw,
    const uint32_t* z_raw,
    uint32_t* out_raw,
    const v2f v_mean,
    const v2f v_rstd) {
  constexpr bool DO_POST_GATE =
      HAS_Z && NORM_BEFORE_GATE && (ACT != GateActivation::kNone);
  constexpr float NEG_LOG2E = -1.44269504088896340736f;
  const v2f v_neg_log2e = {NEG_LOG2E, NEG_LOG2E};
  const v2f v_one = {1.0f, 1.0f};

  v2f z_f[DO_POST_GATE ? N_PAIRS : 1];
  v2f e_f[DO_POST_GATE ? N_PAIRS : 1];
  if constexpr (DO_POST_GATE) {
#pragma unroll
    for (int p = 0; p < N_PAIRS; ++p) {
      z_f[p] = unpack_pair<IS_BF16>(z_raw[p]);
      e_f[p] = z_f[p] * v_neg_log2e;
    }
  }

  v2f y_f[N_PAIRS];
#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    const v2f x_f = unpack_pair<IS_BF16>(x_raw[p]);
    const v2f w_f = unpack_pair<IS_BF16>(w_raw[p]);
    v2f y;
    if constexpr (IS_RMS_NORM) {
      y = x_f * v_rstd;
    } else {
      y = (x_f - v_mean) * v_rstd;
    }
    if constexpr (HAS_BIAS) {
      const v2f b_f = unpack_pair<IS_BF16>(b_raw[p]);
      y = y * w_f + b_f;
    } else {
      y = y * w_f;
    }
    y_f[p] = y;
    if constexpr (DO_POST_GATE) {
      e_f[p] = v2f{__builtin_amdgcn_exp2f(e_f[p][0]),
                   __builtin_amdgcn_exp2f(e_f[p][1])};
    }
  }

  if constexpr (DO_POST_GATE) {
#pragma unroll
    for (int p = 0; p < N_PAIRS; ++p) {
      e_f[p] = v_one + e_f[p];
    }
#pragma unroll
    for (int p = 0; p < N_PAIRS; ++p) {
      e_f[p] = v2f{fast_rcpf(e_f[p][0]), fast_rcpf(e_f[p][1])};
    }
  }

#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    v2f y = y_f[p];
    if constexpr (DO_POST_GATE) {
      if constexpr (ACT == GateActivation::kSilu) {
        y = y * (z_f[p] * e_f[p]);
      } else {
        y = y * e_f[p];
      }
    }
    out_raw[p] = pack_pair_rn<IS_BF16>(y);
  }
}

// Fused Pass-2 epilogue from FP32 v2f x_f (LayerNorm or Mamba2 !NORM_BEFORE_GATE).
template <bool IS_BF16, bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT, int N_PAIRS>
static __device__ __forceinline__ void norm_epilogue_from_v2f(
    const v2f* x_f,
    const uint32_t* w_raw,
    const uint32_t* b_raw,
    const uint32_t* z_raw,
    uint32_t* out_raw,
    const v2f v_mean,
    const v2f v_rstd) {
  constexpr bool DO_POST_GATE =
      HAS_Z && NORM_BEFORE_GATE && (ACT != GateActivation::kNone);
  constexpr float NEG_LOG2E = -1.44269504088896340736f;
  const v2f v_neg_log2e = {NEG_LOG2E, NEG_LOG2E};
  const v2f v_one = {1.0f, 1.0f};

  v2f z_f[DO_POST_GATE ? N_PAIRS : 1];
  v2f e_f[DO_POST_GATE ? N_PAIRS : 1];
  if constexpr (DO_POST_GATE) {
#pragma unroll
    for (int p = 0; p < N_PAIRS; ++p) {
      z_f[p] = unpack_pair<IS_BF16>(z_raw[p]);
      e_f[p] = z_f[p] * v_neg_log2e;
    }
  }

  v2f y_f[N_PAIRS];
#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    const v2f w_f = unpack_pair<IS_BF16>(w_raw[p]);
    v2f y;
    if constexpr (IS_RMS_NORM) {
      y = x_f[p] * v_rstd;
    } else {
      y = (x_f[p] - v_mean) * v_rstd;
    }
    if constexpr (HAS_BIAS) {
      const v2f b_f = unpack_pair<IS_BF16>(b_raw[p]);
      y = y * w_f + b_f;
    } else {
      y = y * w_f;
    }
    y_f[p] = y;
    if constexpr (DO_POST_GATE) {
      e_f[p] = v2f{__builtin_amdgcn_exp2f(e_f[p][0]),
                   __builtin_amdgcn_exp2f(e_f[p][1])};
    }
  }

  if constexpr (DO_POST_GATE) {
#pragma unroll
    for (int p = 0; p < N_PAIRS; ++p) {
      e_f[p] = v_one + e_f[p];
    }
#pragma unroll
    for (int p = 0; p < N_PAIRS; ++p) {
      e_f[p] = v2f{fast_rcpf(e_f[p][0]), fast_rcpf(e_f[p][1])};
    }
  }

#pragma unroll
  for (int p = 0; p < N_PAIRS; ++p) {
    v2f y = y_f[p];
    if constexpr (DO_POST_GATE) {
      if constexpr (ACT == GateActivation::kSilu) {
        y = y * (z_f[p] * e_f[p]);
      } else {
        y = y * e_f[p];
      }
    }
    out_raw[p] = pack_pair_rn<IS_BF16>(y);
  }
}
#endif  // defined(__gfx946__)
}  // namespace ln1p_gfx946

static __device__ __forceinline__ float ln1p_sigmoid(const float x) {
  constexpr float LOG2E = 1.44269504088896340736f;
#if defined(__gfx946__)
  return ln1p_gfx946::fast_rcpf(1.0f + __builtin_amdgcn_exp2f(-x * LOG2E));
#else
  return 1.0f / (1.0f + __builtin_amdgcn_exp2f(-x * LOG2E));
#endif
}

template <GateActivation ACT>
static __device__ __forceinline__ float ln1p_gate(const float z) {
  if constexpr (ACT == GateActivation::kSilu) {
    return z * ln1p_sigmoid(z);
  } else if constexpr (ACT == GateActivation::kSigmoid) {
    return ln1p_sigmoid(z);
  } else {
    return 1.0f;
  }
}

// Adaptive vector load for Z (gate) decoupled from X/Y/W 16B alignment.
// z_vec_mode is a wave-uniform SGPR parameter in {8, 4, 2, 1}, preventing
// non-16B-aligned strided gate views (e.g. Mamba2 projected_states slices)
// from demoting the entire kernel to multi-pass scalar fallback.
template <typename scalar_t, int Vec>
static __device__ __forceinline__ void ln1p_load_z_raw(
    const scalar_t* __restrict__ ptr,
    uint32_t (&dst)[Vec / 2],
    const int z_vec_mode) {
  static_assert(sizeof(scalar_t) == 2 && Vec == 8);
  using Load8 = at::native::memory::aligned_vector<scalar_t, 8>;
  using Load4 = at::native::memory::aligned_vector<scalar_t, 4>;
  using Load2 = at::native::memory::aligned_vector<scalar_t, 2>;
  if (__builtin_expect(z_vec_mode == 8, 1)) {
    const Load8 v = *reinterpret_cast<const Load8*>(ptr);
    __builtin_memcpy(dst, &v, sizeof(dst));
  } else if (z_vec_mode == 4) {
    const auto* src4 = reinterpret_cast<const Load4*>(ptr);
    const Load4 lo = src4[0];
    const Load4 hi = src4[1];
    __builtin_memcpy(dst, &lo, sizeof(Load4));
    __builtin_memcpy(dst + 2, &hi, sizeof(Load4));
  } else if (z_vec_mode == 2) {
    const auto* src2 = reinterpret_cast<const Load2*>(ptr);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const Load2 v = src2[k];
      __builtin_memcpy(&dst[k], &v, sizeof(Load2));
    }
  } else {
    const auto* src1 = reinterpret_cast<const uint16_t*>(ptr);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint32_t lo = static_cast<uint32_t>(src1[p * 2]);
      const uint32_t hi = static_cast<uint32_t>(src1[p * 2 + 1]);
      dst[p] = lo | (hi << 16);
    }
  }
}

template <typename scalar_t, int Vec>
static __device__ __forceinline__ void ln1p_load_z_vec(
    const scalar_t* __restrict__ ptr,
    scalar_t (&dst)[Vec],
    const int z_vec_mode) {
  static_assert(sizeof(scalar_t) == 2 && Vec == 8);
  using Load8 = at::native::memory::aligned_vector<scalar_t, 8>;
  using Load4 = at::native::memory::aligned_vector<scalar_t, 4>;
  using Load2 = at::native::memory::aligned_vector<scalar_t, 2>;
  if (__builtin_expect(z_vec_mode == 8, 1)) {
    *reinterpret_cast<Load8*>(dst) = *reinterpret_cast<const Load8*>(ptr);
  } else if (z_vec_mode == 4) {
    auto* dst4 = reinterpret_cast<Load4*>(dst);
    const auto* src4 = reinterpret_cast<const Load4*>(ptr);
    dst4[0] = src4[0];
    dst4[1] = src4[1];
  } else if (z_vec_mode == 2) {
    auto* dst2 = reinterpret_cast<Load2*>(dst);
    const auto* src2 = reinterpret_cast<const Load2*>(ptr);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      dst2[k] = src2[k];
    }
  } else {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      dst[k] = ptr[k];
    }
  }
}

template <typename T, int reducesize = C10_WARP_SIZE>
__inline__ __device__ T ln1p_warp_reduce_sum(T val) {
#if defined(__gfx946__)
  if constexpr (std::is_same_v<T, float> &&
                (reducesize == 1 || reducesize == 2 || reducesize == 4 ||
                 reducesize == 8 || reducesize == 16 || reducesize == 32 ||
                 reducesize == 64)) {
    return ln1p_gfx946::valu_allreduce_sum_f32<reducesize>(val);
  }
#endif
#pragma unroll
  for (int offset = reducesize / 2; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T, int block_size = 512>
__inline__ __device__ T ln1p_block_reduce_sum(T val, T* shared) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  val = ln1p_warp_reduce_sum<T, C10_WARP_SIZE>(val);

  if constexpr (block_size > C10_WARP_SIZE) {
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0 && wid < share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    // Keep wave 0 EXEC uniform (all 64 lanes active) so DPP/permlane has no
    // VALU->EXEC hazard and lane 0 receives the exact reduced total.
    if (wid == 0) {
      const T lane_val =
          (lid < share_size) ? shared[lid] : static_cast<T>(0);
      val = ln1p_warp_reduce_sum<T, share_size>(lane_val);
    }
  }
  return val;
}

template <typename T>
__inline__ __device__ T ln1p_shfl_xor(T val,
                                      const int lane_mask,
                                      const int width) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  return __shfl_xor(val, lane_mask, width);
#else
  return __shfl_xor_sync(0xffffffff, val, lane_mask, width);
#endif
}

template <typename T, int THREADS_PER_ROW>
__inline__ __device__ T ln1p_row_group_reduce_sum(T val) {
#if defined(__gfx946__)
  if constexpr (std::is_same_v<T, float> &&
                (THREADS_PER_ROW == 8 || THREADS_PER_ROW == 16 ||
                 THREADS_PER_ROW == 32 || THREADS_PER_ROW == 64)) {
    return ln1p_gfx946::valu_allreduce_sum_f32<THREADS_PER_ROW>(val);
  }
#endif
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
    val += ln1p_shfl_xor(val, mask, THREADS_PER_ROW);
  }
  return val;
}

template <typename scalar_t, typename T_ACC, int THREADS_PER_ROW,
          int block_size, int ELEMS_PER_THREAD, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_small_head_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps,
    const T_ACC inv_n) {
  constexpr int ROWS_PER_BLOCK = block_size / THREADS_PER_ROW;
  constexpr int CACHE_ELEMS = ELEMS_PER_THREAD > 0 ? ELEMS_PER_THREAD : 1;

  const int tid = threadIdx.x;
  const int lane = tid % THREADS_PER_ROW;
  const int row_in_block = tid / THREADS_PER_ROW;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;

  for (int64_t row_base = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK;
       row_base < M;
       row_base += static_cast<int64_t>(gridDim.x) * ROWS_PER_BLOCK) {
    const int64_t row = row_base + row_in_block;
    T_ACC local_sum = 0;
    T_ACC local_sq = 0;
    T_ACC x_cache[CACHE_ELEMS];

    if (row < M) {
      int item = 0;
      for (int64_t col = lane; col < N; col += THREADS_PER_ROW) {
        const int64_t x_off = row * stride_x_row + group_col + col;
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          const float z = ln1p_to_float(Z[z_off]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        if constexpr (IS_RMS_NORM) {
          local_sq += x * x;
        } else {
          local_sum += x;
        }
        if constexpr (ELEMS_PER_THREAD > 0) {
          x_cache[item] = x;
        }
        item++;
      }
    }

    T_ACC mean = 0;
    T_ACC var = 0;
    if constexpr (IS_RMS_NORM) {
      local_sq = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_sq);
      var = local_sq * inv_n;
    } else {
      local_sum = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_sum);
      mean = local_sum * inv_n;
      if (lane == 0 && row < M) {
        Mean[group * M + row] = static_cast<float>(mean);
      }
      T_ACC local_var = 0;
      if (row < M) {
        int item = 0;
        for (int64_t col = lane; col < N; col += THREADS_PER_ROW) {
          T_ACC x;
          if constexpr (ELEMS_PER_THREAD > 0) {
            x = x_cache[item];
          } else {
            const int64_t x_off = row * stride_x_row + group_col + col;
            x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
            if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                          ACT != GateActivation::kNone) {
              const int64_t z_off = row * stride_z_row + group_col + col;
              const float z = ln1p_to_float(Z[z_off]);
              x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
            }
          }
          const T_ACC diff = x - mean;
          local_var += diff * diff;
          item++;
        }
      }
      local_var = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_var);
      var = local_var * inv_n;
    }

    const T_ACC rstd = c10::cuda::compat::rsqrt(var + eps);
    if (lane == 0 && row < M) {
      Rstd[group * M + row] = static_cast<float>(rstd);
    }

    if (row < M) {
      int item = 0;
      for (int64_t col = lane; col < N; col += THREADS_PER_ROW) {
        const int64_t y_off = row * stride_y_row + group_col + col;
        const int64_t w_off = group_col + col;

        T_ACC x;
        if constexpr (ELEMS_PER_THREAD > 0) {
          x = x_cache[item];
        } else {
          const int64_t x_off = row * stride_x_row + group_col + col;
          x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
          if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                        ACT != GateActivation::kNone) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            const float z = ln1p_to_float(Z[z_off]);
            x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
          }
        }

        T_ACC y = IS_RMS_NORM ? x * rstd : (x - mean) * rstd;
        y *= static_cast<T_ACC>(ln1p_to_float(W[w_off]));
        if constexpr (HAS_BIAS) {
          y += static_cast<T_ACC>(ln1p_to_float(B[w_off]));
        }
        if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          const float z = ln1p_to_float(Z[z_off]);
          y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }

        scalar_t out;
        ln1p_from_float(out, y);
        Y[y_off] = out;
        item++;
      }
    }
  }
}

template <typename scalar_t, typename T_ACC, int Vec, int THREADS_PER_ROW,
          int block_size, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_small_head_vec_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps,
    const T_ACC inv_n,
    const int z_vec_mode) {
  constexpr int ROWS_PER_BLOCK = block_size / THREADS_PER_ROW;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;

  const int tid = threadIdx.x;
  const int lane = tid % THREADS_PER_ROW;
  const int row_in_block = tid / THREADS_PER_ROW;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;
  const int64_t vec_cols = N / Vec;
  const bool valid_col = lane < vec_cols;
  const int64_t col = static_cast<int64_t>(lane) * Vec;

#if defined(__gfx946__)
  if constexpr (Vec == 8 &&
                (std::is_same_v<scalar_t, at::BFloat16> ||
                 std::is_same_v<scalar_t, at::Half>)) {
    static_assert(std::is_same_v<T_ACC, float>,
                  "BF16/FP16 accumulator type must be float");
    constexpr bool IS_BF16 = std::is_same_v<scalar_t, at::BFloat16>;
    constexpr int Pairs = Vec / 2;
    constexpr bool RAW_RMS_FAST =
        IS_RMS_NORM &&
        (!HAS_Z || NORM_BEFORE_GATE || ACT == GateActivation::kNone);
    constexpr bool DO_POST_GATE =
        HAS_Z && NORM_BEFORE_GATE && (ACT != GateActivation::kNone);
    constexpr bool DO_PRE_GATE =
        HAS_Z && !NORM_BEFORE_GATE && (ACT != GateActivation::kNone);

    // Hoist row-invariant W (and B) loads once per block to overlap with row 0
    // and avoid repeated L1 fetches on multi-row grid-stride loops.
    uint32_t w_raw[Pairs];
    uint32_t b_raw[HAS_BIAS ? Pairs : 1];
    if (valid_col) {
      const int64_t w_off = group_col + col;
      *reinterpret_cast<LoadT*>(w_raw) =
          *reinterpret_cast<const LoadT*>(W + w_off);
      if constexpr (HAS_BIAS) {
        *reinterpret_cast<LoadT*>(b_raw) =
            *reinterpret_cast<const LoadT*>(B + w_off);
      }
    }

    const float eps_f = static_cast<float>(eps);
    const float inv_n_f = static_cast<float>(inv_n);

    for (int64_t row_base = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK;
         row_base < M;
         row_base += static_cast<int64_t>(gridDim.x) * ROWS_PER_BLOCK) {
      const int64_t row = row_base + row_in_block;
      const bool valid = (row < M) && valid_col;

      if constexpr (RAW_RMS_FAST) {
        uint32_t x_raw[Pairs];
        uint32_t z_raw[DO_POST_GATE ? Pairs : 1];
        float local_sq = 0.0f;

        if (valid) {
          const int64_t x_off = row * stride_x_row + group_col + col;
          *reinterpret_cast<LoadT*>(x_raw) =
              *reinterpret_cast<const LoadT*>(X + x_off);
          // Pre-issue Z load right after X so its HBM latency overlaps with
          // dot2 + VALU DPP/permlane reduction + rsqrt (total VGPR <= 64).
          if constexpr (DO_POST_GATE) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_raw, z_vec_mode);
          }
#pragma unroll
          for (int p = 0; p < Pairs; ++p) {
            local_sq = ln1p_gfx946::dot2_sq_pair<IS_BF16>(x_raw[p], local_sq);
          }
        }

        local_sq = ln1p_gfx946::valu_allreduce_sum_f32<THREADS_PER_ROW>(local_sq);
        const float rstd = c10::cuda::compat::rsqrt(local_sq * inv_n_f + eps_f);
        if (lane == 0 && row < M) {
          Rstd[group * M + row] = rstd;
        }

        if (valid) {
          uint32_t out_raw[Pairs];
          const ln1p_gfx946::v2f v_zero = {0.0f, 0.0f};
          const ln1p_gfx946::v2f v_rstd = {rstd, rstd};
          ln1p_gfx946::norm_epilogue_from_raw<IS_BF16, HAS_BIAS, HAS_Z,
                                              NORM_BEFORE_GATE, true, ACT,
                                              Pairs>(
              x_raw, w_raw, ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw),
              ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw), out_raw, v_zero,
              v_rstd);
          const int64_t y_off = row * stride_y_row + group_col + col;
          *reinterpret_cast<LoadT*>(Y + y_off) =
              *reinterpret_cast<const LoadT*>(out_raw);
        }
      } else {
        // LayerNorm (!IS_RMS_NORM) or Mamba2 pre-gate (HAS_Z && !NORM_BEFORE_GATE):
        // Preserve strict FP32 x_f[Pairs] across reduction (never dot2 on gated x).
        ln1p_gfx946::v2f x_f[Pairs];
        uint32_t z_raw[DO_POST_GATE ? Pairs : 1];
        float local_acc = 0.0f;

        if (valid) {
          uint32_t x_raw[Pairs];
          const int64_t x_off = row * stride_x_row + group_col + col;
          *reinterpret_cast<LoadT*>(x_raw) =
              *reinterpret_cast<const LoadT*>(X + x_off);

          uint32_t z_pre[DO_PRE_GATE ? Pairs : 1];
          if constexpr (DO_PRE_GATE) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_pre, z_vec_mode);
          } else if constexpr (DO_POST_GATE) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_raw, z_vec_mode);
          }

#pragma unroll
          for (int p = 0; p < Pairs; ++p) {
            x_f[p] = ln1p_gfx946::unpack_pair<IS_BF16>(x_raw[p]);
          }
          if constexpr (DO_PRE_GATE) {
            ln1p_gfx946::apply_gate_v2f<IS_BF16, ACT, Pairs>(x_f, z_pre);
          }

          ln1p_gfx946::v2f acc_v = {0.0f, 0.0f};
#pragma unroll
          for (int p = 0; p < Pairs; ++p) {
            if constexpr (IS_RMS_NORM) {
              acc_v = acc_v + x_f[p] * x_f[p];  // v_pk_fma_f32
            } else {
              acc_v = acc_v + x_f[p];           // v_pk_add_f32
            }
          }
          local_acc = acc_v[0] + acc_v[1];
        }

        float mean = 0.0f;
        float var = 0.0f;
        if constexpr (IS_RMS_NORM) {
          local_acc =
              ln1p_gfx946::valu_allreduce_sum_f32<THREADS_PER_ROW>(local_acc);
          var = local_acc * inv_n_f;
        } else {
          local_acc =
              ln1p_gfx946::valu_allreduce_sum_f32<THREADS_PER_ROW>(local_acc);
          mean = local_acc * inv_n_f;
          if (lane == 0 && row < M) {
            Mean[group * M + row] = mean;
          }
          float local_var = 0.0f;
          if (valid) {
            const ln1p_gfx946::v2f v_mean = {mean, mean};
            ln1p_gfx946::v2f var_v = {0.0f, 0.0f};
#pragma unroll
            for (int p = 0; p < Pairs; ++p) {
              const ln1p_gfx946::v2f diff = x_f[p] - v_mean;
              var_v = var_v + diff * diff;  // v_pk_fma_f32
            }
            local_var = var_v[0] + var_v[1];
          }
          local_var =
              ln1p_gfx946::valu_allreduce_sum_f32<THREADS_PER_ROW>(local_var);
          var = local_var * inv_n_f;
        }

        const float rstd = c10::cuda::compat::rsqrt(var + eps_f);
        if (lane == 0 && row < M) {
          Rstd[group * M + row] = rstd;
        }

        if (valid) {
          uint32_t out_raw[Pairs];
          const ln1p_gfx946::v2f v_mean = {mean, mean};
          const ln1p_gfx946::v2f v_rstd = {rstd, rstd};
          ln1p_gfx946::norm_epilogue_from_v2f<IS_BF16, HAS_BIAS, HAS_Z,
                                              NORM_BEFORE_GATE, IS_RMS_NORM,
                                              ACT, Pairs>(
              x_f, w_raw, ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw),
              ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw), out_raw, v_mean,
              v_rstd);
          const int64_t y_off = row * stride_y_row + group_col + col;
          *reinterpret_cast<LoadT*>(Y + y_off) =
              *reinterpret_cast<const LoadT*>(out_raw);
        }
      }
    }
    return;
  }
#endif  // defined(__gfx946__)

  scalar_t w_vec[Vec];
  scalar_t b_vec[HAS_BIAS ? Vec : 1];
  if (valid_col) {
    const int64_t w_off = group_col + col;
    *reinterpret_cast<LoadT*>(w_vec) =
        *reinterpret_cast<const LoadT*>(W + w_off);
    if constexpr (HAS_BIAS) {
      *reinterpret_cast<LoadT*>(b_vec) =
          *reinterpret_cast<const LoadT*>(B + w_off);
    }
  }

  for (int64_t row_base = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK;
       row_base < M;
       row_base += static_cast<int64_t>(gridDim.x) * ROWS_PER_BLOCK) {
    const int64_t row = row_base + row_in_block;
    const bool valid = (row < M) && valid_col;
    T_ACC x_acc[Vec];
    T_ACC local = 0;

    if (valid) {
      const int64_t x_off = row * stride_x_row + group_col + col;
      scalar_t x_vec[Vec];
      *reinterpret_cast<LoadT*>(x_vec) =
          *reinterpret_cast<const LoadT*>(X + x_off);

      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                    ACT != GateActivation::kNone) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        ln1p_load_z_vec<scalar_t, Vec>(Z + z_off, z_vec, z_vec_mode);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(x_vec[i]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const float z = ln1p_to_float(z_vec[i]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        x_acc[i] = x;
        if constexpr (IS_RMS_NORM) {
          local += x * x;
        } else {
          local += x;
        }
      }
    }

    T_ACC mean = 0;
    T_ACC var = 0;
    if constexpr (IS_RMS_NORM) {
      local = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local);
      var = local * inv_n;
    } else {
      local = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local);
      mean = local * inv_n;
      if (lane == 0 && row < M) {
        Mean[group * M + row] = static_cast<float>(mean);
      }
      T_ACC local_var = 0;
      if (valid) {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          const T_ACC diff = x_acc[i] - mean;
          local_var += diff * diff;
        }
      }
      local_var = ln1p_row_group_reduce_sum<T_ACC, THREADS_PER_ROW>(local_var);
      var = local_var * inv_n;
    }

    const T_ACC rstd = c10::cuda::compat::rsqrt(var + eps);
    if (lane == 0 && row < M) {
      Rstd[group * M + row] = static_cast<float>(rstd);
    }

    if (valid) {
      const int64_t y_off = row * stride_y_row + group_col + col;
      scalar_t y_vec[Vec];
      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                    ACT != GateActivation::kNone) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        ln1p_load_z_vec<scalar_t, Vec>(Z + z_off, z_vec, z_vec_mode);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC y = IS_RMS_NORM ? x_acc[i] * rstd : (x_acc[i] - mean) * rstd;
        y *= static_cast<T_ACC>(ln1p_to_float(w_vec[i]));
        if constexpr (HAS_BIAS) {
          y += static_cast<T_ACC>(ln1p_to_float(b_vec[i]));
        }
        if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const float z = ln1p_to_float(z_vec[i]);
          y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        ln1p_from_float(y_vec[i], y);
      }
      *reinterpret_cast<LoadT*>(Y + y_off) = *reinterpret_cast<LoadT*>(y_vec);
    }
  }
}

template <typename scalar_t, typename T_ACC, int Vec, int block_size,
          bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_vec_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps,
    const T_ACC inv_n,
    const int z_vec_mode) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;

  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  const int tid = threadIdx.x;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;
  const int64_t vec_cols = N / Vec;
  const bool valid_col = static_cast<int64_t>(tid) < vec_cols;
  const int64_t col = static_cast<int64_t>(tid) * Vec;

#if defined(__gfx946__)
  if constexpr (Vec == 8 &&
                (std::is_same_v<scalar_t, at::BFloat16> ||
                 std::is_same_v<scalar_t, at::Half>)) {
    static_assert(std::is_same_v<T_ACC, float>,
                  "BF16/FP16 accumulator type must be float");
    constexpr bool IS_BF16 = std::is_same_v<scalar_t, at::BFloat16>;
    constexpr int Pairs = Vec / 2;
    constexpr bool RAW_RMS_FAST =
        IS_RMS_NORM &&
        (!HAS_Z || NORM_BEFORE_GATE || ACT == GateActivation::kNone);
    constexpr bool DO_POST_GATE =
        HAS_Z && NORM_BEFORE_GATE && (ACT != GateActivation::kNone);
    constexpr bool DO_PRE_GATE =
        HAS_Z && !NORM_BEFORE_GATE && (ACT != GateActivation::kNone);

    uint32_t w_raw[Pairs];
    uint32_t b_raw[HAS_BIAS ? Pairs : 1];
    if (valid_col) {
      const int64_t w_off = group_col + col;
      *reinterpret_cast<LoadT*>(w_raw) =
          *reinterpret_cast<const LoadT*>(W + w_off);
      if constexpr (HAS_BIAS) {
        *reinterpret_cast<LoadT*>(b_raw) =
            *reinterpret_cast<const LoadT*>(B + w_off);
      }
    }

    const float eps_f = static_cast<float>(eps);
    const float inv_n_f = static_cast<float>(inv_n);

    for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
      if constexpr (RAW_RMS_FAST) {
        uint32_t x_raw[Pairs];
        uint32_t z_raw[DO_POST_GATE ? Pairs : 1];
        float local_sq = 0.0f;

        if (valid_col) {
          const int64_t x_off = row * stride_x_row + group_col + col;
          *reinterpret_cast<LoadT*>(x_raw) =
              *reinterpret_cast<const LoadT*>(X + x_off);
          if constexpr (DO_POST_GATE) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_raw, z_vec_mode);
          }
#pragma unroll
          for (int p = 0; p < Pairs; ++p) {
            local_sq = ln1p_gfx946::dot2_sq_pair<IS_BF16>(x_raw[p], local_sq);
          }
        }

        local_sq = ln1p_block_reduce_sum<float, block_size>(
            local_sq, reinterpret_cast<float*>(shared));
        if (tid == 0) {
          const float rstd_val =
              c10::cuda::compat::rsqrt(local_sq * inv_n_f + eps_f);
          s_rstd = static_cast<T_ACC>(rstd_val);
          Rstd[group * M + row] = rstd_val;
        }
        __syncthreads();

        if (valid_col) {
          const float rstd = static_cast<float>(s_rstd);
          uint32_t out_raw[Pairs];
          const ln1p_gfx946::v2f v_zero = {0.0f, 0.0f};
          const ln1p_gfx946::v2f v_rstd = {rstd, rstd};
          ln1p_gfx946::norm_epilogue_from_raw<IS_BF16, HAS_BIAS, HAS_Z,
                                              NORM_BEFORE_GATE, true, ACT,
                                              Pairs>(
              x_raw, w_raw, ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw),
              ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw), out_raw, v_zero,
              v_rstd);
          const int64_t y_off = row * stride_y_row + group_col + col;
          *reinterpret_cast<LoadT*>(Y + y_off) =
              *reinterpret_cast<const LoadT*>(out_raw);
        }
      } else {
        ln1p_gfx946::v2f x_f[Pairs];
        uint32_t z_raw[DO_POST_GATE ? Pairs : 1];
        float local = 0.0f;

        if (valid_col) {
          uint32_t x_raw[Pairs];
          const int64_t x_off = row * stride_x_row + group_col + col;
          *reinterpret_cast<LoadT*>(x_raw) =
              *reinterpret_cast<const LoadT*>(X + x_off);

          uint32_t z_pre[DO_PRE_GATE ? Pairs : 1];
          if constexpr (DO_PRE_GATE) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_pre, z_vec_mode);
          } else if constexpr (DO_POST_GATE) {
            const int64_t z_off = row * stride_z_row + group_col + col;
            ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_raw, z_vec_mode);
          }

#pragma unroll
          for (int p = 0; p < Pairs; ++p) {
            x_f[p] = ln1p_gfx946::unpack_pair<IS_BF16>(x_raw[p]);
          }
          if constexpr (DO_PRE_GATE) {
            ln1p_gfx946::apply_gate_v2f<IS_BF16, ACT, Pairs>(x_f, z_pre);
          }

          ln1p_gfx946::v2f acc_v = {0.0f, 0.0f};
#pragma unroll
          for (int p = 0; p < Pairs; ++p) {
            if constexpr (IS_RMS_NORM) {
              acc_v = acc_v + x_f[p] * x_f[p];
            } else {
              acc_v = acc_v + x_f[p];
            }
          }
          local = acc_v[0] + acc_v[1];
        }

        local = ln1p_block_reduce_sum<float, block_size>(
            local, reinterpret_cast<float*>(shared));

        if constexpr (IS_RMS_NORM) {
          if (tid == 0) {
            s_mean = static_cast<T_ACC>(0);
            const float rstd_val =
                c10::cuda::compat::rsqrt(local * inv_n_f + eps_f);
            s_rstd = static_cast<T_ACC>(rstd_val);
            Rstd[group * M + row] = rstd_val;
          }
          __syncthreads();
        } else {
          if (tid == 0) {
            const float mean_val = local * inv_n_f;
            s_mean = static_cast<T_ACC>(mean_val);
            Mean[group * M + row] = mean_val;
          }
          __syncthreads();

          const float mean = static_cast<float>(s_mean);
          float var = 0.0f;
          if (valid_col) {
            const ln1p_gfx946::v2f v_mean = {mean, mean};
            ln1p_gfx946::v2f var_v = {0.0f, 0.0f};
#pragma unroll
            for (int p = 0; p < Pairs; ++p) {
              const ln1p_gfx946::v2f diff = x_f[p] - v_mean;
              var_v = var_v + diff * diff;
            }
            var = var_v[0] + var_v[1];
          }
          var = ln1p_block_reduce_sum<float, block_size>(
              var, reinterpret_cast<float*>(shared));
          if (tid == 0) {
            const float rstd_val =
                c10::cuda::compat::rsqrt(var * inv_n_f + eps_f);
            s_rstd = static_cast<T_ACC>(rstd_val);
            Rstd[group * M + row] = rstd_val;
          }
          __syncthreads();
        }

        if (valid_col) {
          const float mean = static_cast<float>(s_mean);
          const float rstd = static_cast<float>(s_rstd);
          uint32_t out_raw[Pairs];
          const ln1p_gfx946::v2f v_mean = {mean, mean};
          const ln1p_gfx946::v2f v_rstd = {rstd, rstd};
          ln1p_gfx946::norm_epilogue_from_v2f<IS_BF16, HAS_BIAS, HAS_Z,
                                              NORM_BEFORE_GATE, IS_RMS_NORM,
                                              ACT, Pairs>(
              x_f, w_raw, ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw),
              ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw), out_raw, v_mean,
              v_rstd);
          const int64_t y_off = row * stride_y_row + group_col + col;
          *reinterpret_cast<LoadT*>(Y + y_off) =
              *reinterpret_cast<const LoadT*>(out_raw);
        }
      }

      if (row + gridDim.x < M) {
        __syncthreads();
      }
    }
    return;
  }
#endif  // defined(__gfx946__)

  scalar_t w_vec[Vec];
  scalar_t b_vec[HAS_BIAS ? Vec : 1];
  if (valid_col) {
    const int64_t w_off = group_col + col;
    *reinterpret_cast<LoadT*>(w_vec) =
        *reinterpret_cast<const LoadT*>(W + w_off);
    if constexpr (HAS_BIAS) {
      *reinterpret_cast<LoadT*>(b_vec) =
          *reinterpret_cast<const LoadT*>(B + w_off);
    }
  }

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    T_ACC x_acc[Vec];
    T_ACC local = 0;

    if (valid_col) {
      const int64_t x_off = row * stride_x_row + group_col + col;
      scalar_t x_vec[Vec];
      *reinterpret_cast<LoadT*>(x_vec) =
          *reinterpret_cast<const LoadT*>(X + x_off);

      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                    ACT != GateActivation::kNone) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        ln1p_load_z_vec<scalar_t, Vec>(Z + z_off, z_vec, z_vec_mode);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(x_vec[i]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const float z = ln1p_to_float(z_vec[i]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        x_acc[i] = x;
        if constexpr (IS_RMS_NORM) {
          local += x * x;
        } else {
          local += x;
        }
      }
    }

    local = ln1p_block_reduce_sum<T_ACC, block_size>(local, shared);

    if constexpr (IS_RMS_NORM) {
      if (tid == 0) {
        s_mean = static_cast<T_ACC>(0);
        s_rstd = c10::cuda::compat::rsqrt(local * inv_n + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    } else {
      if (tid == 0) {
        s_mean = local * inv_n;
        Mean[group * M + row] = static_cast<float>(s_mean);
      }
      __syncthreads();

      const T_ACC mean = s_mean;
      T_ACC var = 0;
      if (valid_col) {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          const T_ACC diff = x_acc[i] - mean;
          var += diff * diff;
        }
      }

      var = ln1p_block_reduce_sum<T_ACC, block_size>(var, shared);
      if (tid == 0) {
        s_rstd = c10::cuda::compat::rsqrt(var * inv_n + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    }

    const T_ACC mean = s_mean;
    const T_ACC rstd = s_rstd;
    if (valid_col) {
      const int64_t y_off = row * stride_y_row + group_col + col;
      scalar_t y_vec[Vec];
      scalar_t z_vec[Vec];
      if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                    ACT != GateActivation::kNone) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        ln1p_load_z_vec<scalar_t, Vec>(Z + z_off, z_vec, z_vec_mode);
      }

#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        T_ACC y = IS_RMS_NORM ? x_acc[i] * rstd : (x_acc[i] - mean) * rstd;
        y *= static_cast<T_ACC>(ln1p_to_float(w_vec[i]));
        if constexpr (HAS_BIAS) {
          y += static_cast<T_ACC>(ln1p_to_float(b_vec[i]));
        }
        if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const float z = ln1p_to_float(z_vec[i]);
          y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        ln1p_from_float(y_vec[i], y);
      }
      *reinterpret_cast<LoadT*>(Y + y_off) = *reinterpret_cast<LoadT*>(y_vec);
    }

    if (row + gridDim.x < M) {
      __syncthreads();
    }
  }
}

template <typename scalar_t, typename T_ACC, int Vec, int block_size,
          int ITEMS_PER_THREAD, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_vec_multi_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps,
    const T_ACC inv_n,
    const int z_vec_mode) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;

  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  const int tid = threadIdx.x;
  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;
  const int64_t vec_cols = N / Vec;

#if defined(__gfx946__)
  if constexpr (Vec == 8 &&
                (std::is_same_v<scalar_t, at::BFloat16> ||
                 std::is_same_v<scalar_t, at::Half>)) {
    static_assert(std::is_same_v<T_ACC, float>,
                  "BF16/FP16 accumulator type must be float");
    constexpr bool IS_BF16 = std::is_same_v<scalar_t, at::BFloat16>;
    constexpr int Pairs = Vec / 2;
    constexpr bool RAW_RMS_FAST =
        IS_RMS_NORM &&
        (!HAS_Z || NORM_BEFORE_GATE || ACT == GateActivation::kNone);
    constexpr bool DO_POST_GATE =
        HAS_Z && NORM_BEFORE_GATE && (ACT != GateActivation::kNone);
    constexpr bool DO_PRE_GATE =
        HAS_Z && !NORM_BEFORE_GATE && (ACT != GateActivation::kNone);

    const float eps_f = static_cast<float>(eps);
    const float inv_n_f = static_cast<float>(inv_n);

    for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
      if constexpr (RAW_RMS_FAST) {
        // Save 8..16 VGPRs vs float x_acc[ITEMS][8] under gfx946 512 KB VGPR/CU.
        uint32_t x_raw[ITEMS_PER_THREAD][Pairs];
        float local_sq = 0.0f;

#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
          const int64_t vec_col =
              static_cast<int64_t>(tid) + static_cast<int64_t>(item) * block_size;
          if (vec_col < vec_cols) {
            const int64_t col = vec_col * Vec;
            const int64_t x_off = row * stride_x_row + group_col + col;
            *reinterpret_cast<LoadT*>(x_raw[item]) =
                *reinterpret_cast<const LoadT*>(X + x_off);
#pragma unroll
            for (int p = 0; p < Pairs; ++p) {
              local_sq =
                  ln1p_gfx946::dot2_sq_pair<IS_BF16>(x_raw[item][p], local_sq);
            }
          }
        }

        local_sq = ln1p_block_reduce_sum<float, block_size>(
            local_sq, reinterpret_cast<float*>(shared));
        if (tid == 0) {
          const float rstd_val =
              c10::cuda::compat::rsqrt(local_sq * inv_n_f + eps_f);
          s_rstd = static_cast<T_ACC>(rstd_val);
          Rstd[group * M + row] = rstd_val;
        }
        __syncthreads();

        const float rstd = static_cast<float>(s_rstd);
        const ln1p_gfx946::v2f v_zero = {0.0f, 0.0f};
        const ln1p_gfx946::v2f v_rstd = {rstd, rstd};

        // Process on-demand W/B/Z in 2-pair (4-element) chunks per item to keep
        // peak VGPR <= 64 (8 waves/SIMD, SCRATCH=0) while still clearing §4.2.3 SPF hazards.
#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
          const int64_t vec_col =
              static_cast<int64_t>(tid) + static_cast<int64_t>(item) * block_size;
          if (vec_col < vec_cols) {
            const int64_t col = vec_col * Vec;
            const int64_t y_off = row * stride_y_row + group_col + col;
            const int64_t w_off = group_col + col;
            uint32_t w_raw[Pairs];
            uint32_t b_raw[HAS_BIAS ? Pairs : 1];
            uint32_t z_raw[DO_POST_GATE ? Pairs : 1];
            uint32_t out_raw[Pairs];

            *reinterpret_cast<LoadT*>(w_raw) =
                *reinterpret_cast<const LoadT*>(W + w_off);
            if constexpr (HAS_BIAS) {
              *reinterpret_cast<LoadT*>(b_raw) =
                  *reinterpret_cast<const LoadT*>(B + w_off);
            }
            if constexpr (DO_POST_GATE) {
              const int64_t z_off = row * stride_z_row + group_col + col;
              ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_raw, z_vec_mode);
            }

            ln1p_gfx946::norm_epilogue_from_raw<IS_BF16, HAS_BIAS, HAS_Z,
                                                NORM_BEFORE_GATE, true, ACT, 2>(
                x_raw[item], w_raw, ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw, 0),
                ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw, 0), out_raw, v_zero,
                v_rstd);
            ln1p_gfx946::norm_epilogue_from_raw<IS_BF16, HAS_BIAS, HAS_Z,
                                                NORM_BEFORE_GATE, true, ACT, 2>(
                x_raw[item] + 2, w_raw + 2,
                ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw, 2),
                ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw, 2), out_raw + 2,
                v_zero, v_rstd);

            *reinterpret_cast<LoadT*>(Y + y_off) =
                *reinterpret_cast<const LoadT*>(out_raw);
          }
        }
      } else {
        ln1p_gfx946::v2f x_f[ITEMS_PER_THREAD][Pairs];
        ln1p_gfx946::v2f acc_v = {0.0f, 0.0f};

#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
          const int64_t vec_col =
              static_cast<int64_t>(tid) + static_cast<int64_t>(item) * block_size;
          if (vec_col < vec_cols) {
            const int64_t col = vec_col * Vec;
            const int64_t x_off = row * stride_x_row + group_col + col;
            uint32_t x_raw[Pairs];
            *reinterpret_cast<LoadT*>(x_raw) =
                *reinterpret_cast<const LoadT*>(X + x_off);

            uint32_t z_pre[DO_PRE_GATE ? Pairs : 1];
            if constexpr (DO_PRE_GATE) {
              const int64_t z_off = row * stride_z_row + group_col + col;
              ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_pre, z_vec_mode);
            }

#pragma unroll
            for (int c = 0; c < 2; ++c) {
              ln1p_gfx946::v2f chunk[2];
              chunk[0] = ln1p_gfx946::unpack_pair<IS_BF16>(x_raw[c * 2]);
              chunk[1] = ln1p_gfx946::unpack_pair<IS_BF16>(x_raw[c * 2 + 1]);
              if constexpr (DO_PRE_GATE) {
                ln1p_gfx946::apply_gate_v2f<IS_BF16, ACT, 2>(chunk,
                                                             z_pre + c * 2);
              }
              x_f[item][c * 2] = chunk[0];
              x_f[item][c * 2 + 1] = chunk[1];
              if constexpr (IS_RMS_NORM) {
                acc_v = acc_v + chunk[0] * chunk[0];
                acc_v = acc_v + chunk[1] * chunk[1];
              } else {
                acc_v = acc_v + chunk[0];
                acc_v = acc_v + chunk[1];
              }
            }
          }
        }

        float local = acc_v[0] + acc_v[1];
        local = ln1p_block_reduce_sum<float, block_size>(
            local, reinterpret_cast<float*>(shared));

        if constexpr (IS_RMS_NORM) {
          if (tid == 0) {
            s_mean = static_cast<T_ACC>(0);
            const float rstd_val =
                c10::cuda::compat::rsqrt(local * inv_n_f + eps_f);
            s_rstd = static_cast<T_ACC>(rstd_val);
            Rstd[group * M + row] = rstd_val;
          }
          __syncthreads();
        } else {
          if (tid == 0) {
            const float mean_val = local * inv_n_f;
            s_mean = static_cast<T_ACC>(mean_val);
            Mean[group * M + row] = mean_val;
          }
          __syncthreads();

          const float mean = static_cast<float>(s_mean);
          const ln1p_gfx946::v2f v_mean = {mean, mean};
          ln1p_gfx946::v2f var_v = {0.0f, 0.0f};
#pragma unroll
          for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const int64_t vec_col =
                static_cast<int64_t>(tid) +
                static_cast<int64_t>(item) * block_size;
            if (vec_col < vec_cols) {
#pragma unroll
              for (int p = 0; p < Pairs; ++p) {
                const ln1p_gfx946::v2f diff = x_f[item][p] - v_mean;
                var_v = var_v + diff * diff;
              }
            }
          }
          float var = var_v[0] + var_v[1];
          var = ln1p_block_reduce_sum<float, block_size>(
              var, reinterpret_cast<float*>(shared));
          if (tid == 0) {
            const float rstd_val =
                c10::cuda::compat::rsqrt(var * inv_n_f + eps_f);
            s_rstd = static_cast<T_ACC>(rstd_val);
            Rstd[group * M + row] = rstd_val;
          }
          __syncthreads();
        }

        const float mean = static_cast<float>(s_mean);
        const float rstd = static_cast<float>(s_rstd);
        const ln1p_gfx946::v2f v_mean = {mean, mean};
        const ln1p_gfx946::v2f v_rstd = {rstd, rstd};

#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
          const int64_t vec_col =
              static_cast<int64_t>(tid) + static_cast<int64_t>(item) * block_size;
          if (vec_col < vec_cols) {
            const int64_t col = vec_col * Vec;
            const int64_t y_off = row * stride_y_row + group_col + col;
            const int64_t w_off = group_col + col;
            uint32_t w_raw[Pairs];
            uint32_t b_raw[HAS_BIAS ? Pairs : 1];
            uint32_t z_raw[DO_POST_GATE ? Pairs : 1];
            uint32_t out_raw[Pairs];

            *reinterpret_cast<LoadT*>(w_raw) =
                *reinterpret_cast<const LoadT*>(W + w_off);
            if constexpr (HAS_BIAS) {
              *reinterpret_cast<LoadT*>(b_raw) =
                  *reinterpret_cast<const LoadT*>(B + w_off);
            }
            if constexpr (DO_POST_GATE) {
              const int64_t z_off = row * stride_z_row + group_col + col;
              ln1p_load_z_raw<scalar_t, Vec>(Z + z_off, z_raw, z_vec_mode);
            }

            ln1p_gfx946::norm_epilogue_from_v2f<IS_BF16, HAS_BIAS, HAS_Z,
                                                NORM_BEFORE_GATE, IS_RMS_NORM,
                                                ACT, 2>(
                x_f[item], w_raw, ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw, 0),
                ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw, 0), out_raw, v_mean,
                v_rstd);
            ln1p_gfx946::norm_epilogue_from_v2f<IS_BF16, HAS_BIAS, HAS_Z,
                                                NORM_BEFORE_GATE, IS_RMS_NORM,
                                                ACT, 2>(
                x_f[item] + 2, w_raw + 2,
                ln1p_gfx946::opt_ptr<HAS_BIAS>(b_raw, 2),
                ln1p_gfx946::opt_ptr<DO_POST_GATE>(z_raw, 2), out_raw + 2,
                v_mean, v_rstd);

            *reinterpret_cast<LoadT*>(Y + y_off) =
                *reinterpret_cast<const LoadT*>(out_raw);
          }
        }
      }

      if (row + gridDim.x < M) {
        __syncthreads();
      }
    }
    return;
  }
#endif  // defined(__gfx946__)

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    T_ACC x_acc[ITEMS_PER_THREAD][Vec];
    T_ACC local = 0;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int64_t vec_col =
          static_cast<int64_t>(tid) + static_cast<int64_t>(item) * block_size;
      if (vec_col < vec_cols) {
        const int64_t col = vec_col * Vec;
        const int64_t x_off = row * stride_x_row + group_col + col;
        scalar_t x_vec[Vec];
        *reinterpret_cast<LoadT*>(x_vec) =
            *reinterpret_cast<const LoadT*>(X + x_off);

        scalar_t z_vec[Vec];
        if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          ln1p_load_z_vec<scalar_t, Vec>(Z + z_off, z_vec, z_vec_mode);
        }

#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          T_ACC x = static_cast<T_ACC>(ln1p_to_float(x_vec[i]));
          if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                        ACT != GateActivation::kNone) {
            const float z = ln1p_to_float(z_vec[i]);
            x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
          }
          x_acc[item][i] = x;
          if constexpr (IS_RMS_NORM) {
            local += x * x;
          } else {
            local += x;
          }
        }
      }
    }

    local = ln1p_block_reduce_sum<T_ACC, block_size>(local, shared);

    if constexpr (IS_RMS_NORM) {
      if (tid == 0) {
        s_mean = static_cast<T_ACC>(0);
        s_rstd = c10::cuda::compat::rsqrt(local * inv_n + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    } else {
      if (tid == 0) {
        s_mean = local * inv_n;
        Mean[group * M + row] = static_cast<float>(s_mean);
      }
      __syncthreads();

      const T_ACC mean = s_mean;
      T_ACC var = 0;
#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int64_t vec_col =
            static_cast<int64_t>(tid) + static_cast<int64_t>(item) * block_size;
        if (vec_col < vec_cols) {
#pragma unroll
          for (int i = 0; i < Vec; ++i) {
            const T_ACC diff = x_acc[item][i] - mean;
            var += diff * diff;
          }
        }
      }

      var = ln1p_block_reduce_sum<T_ACC, block_size>(var, shared);
      if (tid == 0) {
        s_rstd = c10::cuda::compat::rsqrt(var * inv_n + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    }

    const T_ACC mean = s_mean;
    const T_ACC rstd = s_rstd;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int64_t vec_col =
          static_cast<int64_t>(tid) + static_cast<int64_t>(item) * block_size;
      if (vec_col < vec_cols) {
        const int64_t col = vec_col * Vec;
        const int64_t y_off = row * stride_y_row + group_col + col;
        const int64_t w_off = group_col + col;

        scalar_t w_vec[Vec];
        scalar_t y_vec[Vec];
        *reinterpret_cast<LoadT*>(w_vec) =
            *reinterpret_cast<const LoadT*>(W + w_off);

        scalar_t b_vec[Vec];
        if constexpr (HAS_BIAS) {
          *reinterpret_cast<LoadT*>(b_vec) =
              *reinterpret_cast<const LoadT*>(B + w_off);
        }

        scalar_t z_vec[Vec];
        if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          ln1p_load_z_vec<scalar_t, Vec>(Z + z_off, z_vec, z_vec_mode);
        }

#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          T_ACC y =
              IS_RMS_NORM ? x_acc[item][i] * rstd : (x_acc[item][i] - mean) * rstd;
          y *= static_cast<T_ACC>(ln1p_to_float(w_vec[i]));
          if constexpr (HAS_BIAS) {
            y += static_cast<T_ACC>(ln1p_to_float(b_vec[i]));
          }
          if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                        ACT != GateActivation::kNone) {
            const float z = ln1p_to_float(z_vec[i]);
            y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
          }
          ln1p_from_float(y_vec[i], y);
        }
        *reinterpret_cast<LoadT*>(Y + y_off) = *reinterpret_cast<LoadT*>(y_vec);
      }
    }

    if (row + gridDim.x < M) {
      __syncthreads();
    }
  }
}

template <typename scalar_t, typename T_ACC, int block_size,
          bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT>
__global__ void layer_norm_fwd_1pass_scalar_kernel(
    const scalar_t* X,
    scalar_t* Y,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ Z,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const T_ACC eps,
    const T_ACC inv_n) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC shared[share_size];
  __shared__ T_ACC s_mean;
  __shared__ T_ACC s_rstd;

  const int64_t group = blockIdx.y;
  const int64_t group_col = group * N;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    T_ACC local = 0;
    for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
      const int64_t x_off = row * stride_x_row + group_col + col;
      T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
      if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                    ACT != GateActivation::kNone) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        const float z = ln1p_to_float(Z[z_off]);
        x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
      }
      if constexpr (IS_RMS_NORM) {
        local += x * x;
      } else {
        local += x;
      }
    }

    local = ln1p_block_reduce_sum<T_ACC, block_size>(local, shared);

    if constexpr (IS_RMS_NORM) {
      if (threadIdx.x == 0) {
        s_mean = static_cast<T_ACC>(0);
        s_rstd = c10::cuda::compat::rsqrt(local * inv_n + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        s_mean = local * inv_n;
        Mean[group * M + row] = static_cast<float>(s_mean);
      }
      __syncthreads();

      const T_ACC mean = s_mean;
      T_ACC var = 0;
      for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
        const int64_t x_off = row * stride_x_row + group_col + col;
        T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
        if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                      ACT != GateActivation::kNone) {
          const int64_t z_off = row * stride_z_row + group_col + col;
          const float z = ln1p_to_float(Z[z_off]);
          x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
        }
        const T_ACC diff = x - mean;
        var += diff * diff;
      }

      var = ln1p_block_reduce_sum<T_ACC, block_size>(var, shared);
      if (threadIdx.x == 0) {
        s_rstd = c10::cuda::compat::rsqrt(var * inv_n + eps);
        Rstd[group * M + row] = static_cast<float>(s_rstd);
      }
      __syncthreads();
    }

    const T_ACC mean = s_mean;
    const T_ACC rstd = s_rstd;
    for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
      const int64_t x_off = row * stride_x_row + group_col + col;
      const int64_t y_off = row * stride_y_row + group_col + col;
      const int64_t w_off = group_col + col;

      T_ACC x = static_cast<T_ACC>(ln1p_to_float(X[x_off]));
      if constexpr (HAS_Z && !NORM_BEFORE_GATE &&
                    ACT != GateActivation::kNone) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        const float z = ln1p_to_float(Z[z_off]);
        x *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
      }

      T_ACC y = IS_RMS_NORM ? x * rstd : (x - mean) * rstd;
      y *= static_cast<T_ACC>(ln1p_to_float(W[w_off]));
      if constexpr (HAS_BIAS) {
        y += static_cast<T_ACC>(ln1p_to_float(B[w_off]));
      }
      if constexpr (HAS_Z && NORM_BEFORE_GATE &&
                    ACT != GateActivation::kNone) {
        const int64_t z_off = row * stride_z_row + group_col + col;
        const float z = ln1p_to_float(Z[z_off]);
        y *= static_cast<T_ACC>(ln1p_gate<ACT>(z));
      }

      scalar_t out;
      ln1p_from_float(out, y);
      Y[y_off] = out;
    }

    if (row + gridDim.x < M) {
      __syncthreads();
    }
  }
}

template <typename scalar_t, int Vec>
static bool ln1p_can_vectorize_base(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const torch::Tensor& W,
    const torch::Tensor& B,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t N,
    const bool HAS_BIAS) {
  const int64_t bytes = static_cast<int64_t>(sizeof(scalar_t)) * Vec;
  const auto aligned = [bytes](const torch::Tensor& t) {
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % bytes == 0;
  };

  if (N % Vec != 0 || stride_x_row % Vec != 0 || stride_y_row % Vec != 0) {
    return false;
  }
  if (!aligned(X) || !aligned(Y) || !aligned(W)) {
    return false;
  }
  if (HAS_BIAS && !aligned(B)) {
    return false;
  }
  return true;
}

template <typename scalar_t>
static int ln1p_z_vec_mode(
    const torch::Tensor& Z,
    const int64_t stride_z_row,
    const int64_t N,
    const bool HAS_Z,
    const int max_vec) {
  if (!HAS_Z) {
    return max_vec;
  }
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(Z.data_ptr());
  for (int v = max_vec; v >= 2; v >>= 1) {
    const int64_t bytes = static_cast<int64_t>(sizeof(scalar_t)) * v;
    if (N % v == 0 && stride_z_row % v == 0 && (addr % bytes == 0)) {
      return v;
    }
  }
  return 1;
}

static int ln1p_get_cu_count() {
  int cu_count = 1;
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop != nullptr && prop->multiProcessorCount > 0) {
    cu_count = prop->multiProcessorCount;
  }
  return cu_count;
}

static int ln1p_small_head_grid_x(const int64_t M,
                                  const int rows_per_block,
                                  const int64_t groups,
                                  const int cu_count) {
  const int64_t row_blocks = (M + rows_per_block - 1) / rows_per_block;
  TORCH_CHECK(row_blocks > 0, "row_blocks must be positive");
  int64_t grid_x = row_blocks;
  if (groups == 1 && row_blocks >= static_cast<int64_t>(cu_count) * 4) {
    grid_x = (row_blocks / cu_count) * cu_count;
    grid_x = std::max<int64_t>(grid_x, static_cast<int64_t>(cu_count));
  }
  return static_cast<int>(std::min<int64_t>(grid_x, 65535));
}

template <typename scalar_t, typename T_ACC, int THREADS_PER_ROW,
          bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE, bool IS_RMS_NORM,
          GateActivation ACT>
static void ln1p_launch_small_head(
    const scalar_t* x_data,
    scalar_t* y_data,
    const scalar_t* w_data,
    const scalar_t* b_data,
    const scalar_t* z_data,
    float* mean_data,
    float* rstd_data,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const int64_t groups,
    const T_ACC eps,
    const T_ACC inv_n,
    const bool can_vec8,
    const int z_vec_mode,
    const int cu_count,
    const cudaStream_t stream) {
  // Small/medium M uses 1-wave blocks (64 threads) so work spreads across all
  // 192 CUs (4x48 XCDs on gfx946); large M uses 2-wave blocks (128 threads).
  const bool use_block64 =
      (M * groups <= static_cast<int64_t>(cu_count) * 16);

  if (use_block64) {
    constexpr int small_block = 64;
    constexpr int rows_per_block = small_block / THREADS_PER_ROW;
    const int grid_x =
        ln1p_small_head_grid_x(M, rows_per_block, groups, cu_count);
    dim3 grid(grid_x, static_cast<unsigned int>(groups));
    if constexpr (sizeof(scalar_t) == 2) {
      if (can_vec8) {
        layer_norm_fwd_1pass_small_head_vec_kernel<
            scalar_t, T_ACC, 8, THREADS_PER_ROW, small_block, HAS_BIAS, HAS_Z,
            NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
            <<<grid, small_block, 0, stream>>>(
                x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
                stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n,
                z_vec_mode);
        return;
      }
    }
    layer_norm_fwd_1pass_small_head_kernel<
        scalar_t, T_ACC, THREADS_PER_ROW, small_block, 8, HAS_BIAS, HAS_Z,
        NORM_BEFORE_GATE, IS_RMS_NORM, ACT><<<grid, small_block, 0, stream>>>(
        x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
        stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n);
  } else {
    constexpr int small_block = 128;
    constexpr int rows_per_block = small_block / THREADS_PER_ROW;
    const int grid_x =
        ln1p_small_head_grid_x(M, rows_per_block, groups, cu_count);
    dim3 grid(grid_x, static_cast<unsigned int>(groups));
    if constexpr (sizeof(scalar_t) == 2) {
      if (can_vec8) {
        layer_norm_fwd_1pass_small_head_vec_kernel<
            scalar_t, T_ACC, 8, THREADS_PER_ROW, small_block, HAS_BIAS, HAS_Z,
            NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
            <<<grid, small_block, 0, stream>>>(
                x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
                stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n,
                z_vec_mode);
        return;
      }
    }
    layer_norm_fwd_1pass_small_head_kernel<
        scalar_t, T_ACC, THREADS_PER_ROW, small_block, 8, HAS_BIAS, HAS_Z,
        NORM_BEFORE_GATE, IS_RMS_NORM, ACT><<<grid, small_block, 0, stream>>>(
        x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
        stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n);
  }
}

template <typename scalar_t, bool HAS_BIAS, bool HAS_Z,
          bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT>
static void ln1p_launch_typed(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    torch::Tensor& B,
    torch::Tensor& Z,
    torch::Tensor& Mean,
    torch::Tensor& Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const int64_t groups,
    const double epsilon,
    const cudaStream_t stream) {
  using T_ACC = at::acc_type<scalar_t, true>;
  const T_ACC eps = static_cast<T_ACC>(epsilon);
  const T_ACC inv_n = static_cast<T_ACC>(1.0) / static_cast<T_ACC>(N);
  const int cu_count = ln1p_get_cu_count();

  scalar_t* x_data = X.data_ptr<scalar_t>();
  scalar_t* y_data = Y.data_ptr<scalar_t>();
  scalar_t* w_data = W.data_ptr<scalar_t>();
  scalar_t* b_data = HAS_BIAS ? B.data_ptr<scalar_t>() : nullptr;
  scalar_t* z_data = HAS_Z ? Z.data_ptr<scalar_t>() : nullptr;
  float* mean_data =
      (!IS_RMS_NORM && Mean.defined()) ? Mean.data_ptr<float>() : nullptr;
  float* rstd_data = Rstd.data_ptr<float>();

  const bool can_vec8 =
      (sizeof(scalar_t) == 2) &&
      ln1p_can_vectorize_base<scalar_t, 8>(
          X, Y, W, B, stride_x_row, stride_y_row, N, HAS_BIAS);
  const int z_vec_mode =
      can_vec8 ? ln1p_z_vec_mode<scalar_t>(Z, stride_z_row, N, HAS_Z, 8) : 1;

  if (N <= 512) {
    if (N <= 64) {
      ln1p_launch_small_head<scalar_t, T_ACC, 8, HAS_BIAS, HAS_Z,
                             NORM_BEFORE_GATE, IS_RMS_NORM, ACT>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, groups, eps, inv_n,
          can_vec8, z_vec_mode, cu_count, stream);
      return;
    }
    if (N <= 128) {
      ln1p_launch_small_head<scalar_t, T_ACC, 16, HAS_BIAS, HAS_Z,
                             NORM_BEFORE_GATE, IS_RMS_NORM, ACT>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, groups, eps, inv_n,
          can_vec8, z_vec_mode, cu_count, stream);
      return;
    }
    if (N <= 256) {
      ln1p_launch_small_head<scalar_t, T_ACC, 32, HAS_BIAS, HAS_Z,
                             NORM_BEFORE_GATE, IS_RMS_NORM, ACT>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, groups, eps, inv_n,
          can_vec8, z_vec_mode, cu_count, stream);
      return;
    }
    ln1p_launch_small_head<scalar_t, T_ACC, 64, HAS_BIAS, HAS_Z,
                           NORM_BEFORE_GATE, IS_RMS_NORM, ACT>(
        x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
        stride_x_row, stride_y_row, stride_z_row, M, N, groups, eps, inv_n,
        can_vec8, z_vec_mode, cu_count, stream);
    return;
  }

  int grid_x = static_cast<int>(std::min<int64_t>(M, 65535));
  if (M == 288 && cu_count <= 128) {
    grid_x = 256;
  }
  dim3 grid(grid_x, static_cast<unsigned int>(groups));

  if constexpr (sizeof(scalar_t) == 2) {
    if (can_vec8 && N <= 16384) {
      if (N <= 1024) {
        layer_norm_fwd_1pass_vec_kernel<scalar_t, T_ACC, 8, 128, HAS_BIAS,
                                        HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                                        ACT><<<grid, 128, 0, stream>>>(
            x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
            stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n,
            z_vec_mode);
        return;
      }
      if (N <= 2048) {
        layer_norm_fwd_1pass_vec_kernel<scalar_t, T_ACC, 8, 256, HAS_BIAS,
                                        HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                                        ACT><<<grid, 256, 0, stream>>>(
            x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
            stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n,
            z_vec_mode);
        return;
      }
      if (N <= 4096) {
        layer_norm_fwd_1pass_vec_multi_kernel<
            scalar_t, T_ACC, 8, 256, 2, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
            IS_RMS_NORM, ACT><<<grid, 256, 0, stream>>>(
            x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
            stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n,
            z_vec_mode);
        return;
      }
      if (N <= 8192) {
        layer_norm_fwd_1pass_vec_multi_kernel<
            scalar_t, T_ACC, 8, 256, 4, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
            IS_RMS_NORM, ACT><<<grid, 256, 0, stream>>>(
            x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
            stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n,
            z_vec_mode);
        return;
      }
      layer_norm_fwd_1pass_vec_multi_kernel<
          scalar_t, T_ACC, 8, 512, 4, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
          IS_RMS_NORM, ACT><<<grid, 512, 0, stream>>>(
          x_data, y_data, w_data, b_data, z_data, mean_data, rstd_data,
          stride_x_row, stride_y_row, stride_z_row, M, N, eps, inv_n,
          z_vec_mode);
      return;
    }
  }

  if (N <= 1024) {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 128, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 128, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                   mean_data, rstd_data, stride_x_row,
                                   stride_y_row, stride_z_row, M, N, eps,
                                   inv_n);
  } else if (N <= 2048) {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 256, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 256, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                   mean_data, rstd_data, stride_x_row,
                                   stride_y_row, stride_z_row, M, N, eps,
                                   inv_n);
  } else if (N <= 4096) {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 512, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 512, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                   mean_data, rstd_data, stride_x_row,
                                   stride_y_row, stride_z_row, M, N, eps,
                                   inv_n);
  } else {
    layer_norm_fwd_1pass_scalar_kernel<scalar_t, T_ACC, 1024, HAS_BIAS, HAS_Z,
                                       NORM_BEFORE_GATE, IS_RMS_NORM, ACT>
        <<<grid, 1024, 0, stream>>>(x_data, y_data, w_data, b_data, z_data,
                                    mean_data, rstd_data, stride_x_row,
                                    stride_y_row, stride_z_row, M, N, eps,
                                    inv_n);
  }
}

template <bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM, GateActivation ACT>
static void ln1p_dispatch_dtype(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    torch::Tensor& B,
    torch::Tensor& Z,
    torch::Tensor& Mean,
    torch::Tensor& Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const int64_t groups,
    const double eps,
    const cudaStream_t stream) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "layer_norm_fwd_1pass_opt",
      [&] {
        ln1p_launch_typed<scalar_t, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE,
                          IS_RMS_NORM, ACT>(
            X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row,
            stride_z_row, M, N, groups, eps, stream);
      });
}

template <bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE,
          bool IS_RMS_NORM>
static void ln1p_dispatch_activation(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    torch::Tensor& B,
    torch::Tensor& Z,
    torch::Tensor& Mean,
    torch::Tensor& Rstd,
    const int64_t stride_x_row,
    const int64_t stride_y_row,
    const int64_t stride_z_row,
    const int64_t M,
    const int64_t N,
    const int64_t groups,
    const double eps,
    const GateActivation activation,
    const cudaStream_t stream) {
  if (!HAS_Z || activation == GateActivation::kNone) {
    ln1p_dispatch_dtype<HAS_BIAS, HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                        GateActivation::kNone>(
        X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,
        N, groups, eps, stream);
  } else if (activation == GateActivation::kSilu) {
    ln1p_dispatch_dtype<HAS_BIAS, HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                        GateActivation::kSilu>(
        X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,
        N, groups, eps, stream);
  } else {
    ln1p_dispatch_dtype<HAS_BIAS, HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM,
                        GateActivation::kSigmoid>(
        X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,
        N, groups, eps, stream);
  }
}

static GateActivation ln1p_parse_activation(const std::string& activation) {
  if (activation == "swish" || activation == "silu") {
    return GateActivation::kSilu;
  }
  if (activation == "sigmoid") {
    return GateActivation::kSigmoid;
  }
  if (activation.empty() || activation == "none" || activation == "identity") {
    return GateActivation::kNone;
  }
  TORCH_CHECK(false,
              "Unsupported ACTIVATION for layer_norm_fwd_1pass_opt: ",
              activation,
              ". Expected 'swish'/'silu', 'sigmoid', or 'none'.");
  return GateActivation::kNone;
}

static void ln1p_check_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

static void ln1p_check_same_device(const torch::Tensor& t,
                                   const torch::Tensor& ref,
                                   const char* name) {
  TORCH_CHECK(t.device() == ref.device(),
              name,
              " must be on the same CUDA device as X");
}

// Validate strided 2D/flattened views (e.g. Mamba2 gate = projected_states.split)
// against backing storage capacity from storage_offset() instead of logical numel().
static void ln1p_check_strided_capacity(const torch::Tensor& t,
                                        const int64_t M,
                                        const int64_t stride_row,
                                        const int64_t row_elems,
                                        const char* name) {
  TORCH_CHECK(t.has_storage(), name, " must have backing storage");
  if (t.dim() > 0 && t.size(-1) > 1) {
    TORCH_CHECK(t.stride(-1) == 1,
                name,
                " must be contiguous along the last dimension");
  }
  TORCH_CHECK(t.numel() >= M * row_elems,
              name,
              " does not contain enough logical elements for M/groups/N");
  const int64_t elem_size = static_cast<int64_t>(t.element_size());
  const int64_t storage_elems =
      static_cast<int64_t>(t.storage().nbytes()) / elem_size;
  const int64_t avail_elems = storage_elems - t.storage_offset();
  const int64_t required = (M - 1) * stride_row + row_elems;
  TORCH_CHECK(
      avail_elems >= required,
      name,
      " storage does not contain enough elements for M/stride_row/groups/N");
}

void layer_norm_fwd_1pass_opt(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    std::optional<torch::Tensor> B_opt,
    std::optional<torch::Tensor> Z_opt,
    std::optional<torch::Tensor> Mean_opt,
    torch::Tensor& Rstd,
    int64_t stride_x_row,
    int64_t stride_y_row,
    int64_t stride_z_row,
    int64_t M,
    int64_t N,
    double eps,
    int64_t BLOCK_N,
    int64_t ROWS_PER_BLOCK,
    bool HAS_BIAS,
    bool HAS_Z,
    bool NORM_BEFORE_GATE,
    bool IS_RMS_NORM,
    const std::string& ACTIVATION) {
  TORCH_CHECK(M >= 0, "M must be non-negative");
  TORCH_CHECK(N > 0, "N must be positive");
  TORCH_CHECK(BLOCK_N >= N,
              "BLOCK_N must cover N, matching the Triton kernel assumption");
  TORCH_CHECK(ROWS_PER_BLOCK >= 1, "ROWS_PER_BLOCK must be positive");

  if (M == 0) {
    return;
  }

  torch::Tensor B = B_opt.has_value() ? *B_opt : torch::Tensor();
  torch::Tensor Z = Z_opt.has_value() ? *Z_opt : torch::Tensor();
  torch::Tensor Mean = Mean_opt.has_value() ? *Mean_opt : torch::Tensor();

  ln1p_check_tensor(X, "X");
  ln1p_check_tensor(Y, "Y");
  ln1p_check_tensor(W, "W");
  ln1p_check_tensor(Rstd, "Rstd");
  ln1p_check_same_device(Y, X, "Y");
  ln1p_check_same_device(W, X, "W");
  ln1p_check_same_device(Rstd, X, "Rstd");
  if (HAS_BIAS) {
    ln1p_check_tensor(B, "B");
    ln1p_check_same_device(B, X, "B");
  }
  if (HAS_Z) {
    ln1p_check_tensor(Z, "Z");
    ln1p_check_same_device(Z, X, "Z");
  }
  if (!IS_RMS_NORM) {
    ln1p_check_tensor(Mean, "Mean");
    ln1p_check_same_device(Mean, X, "Mean");
  }

  TORCH_CHECK(Y.scalar_type() == X.scalar_type(),
              "Y must have the same dtype as X");
  TORCH_CHECK(W.scalar_type() == X.scalar_type(),
              "W must have the same dtype as X");
  if (HAS_BIAS) {
    TORCH_CHECK(B.scalar_type() == X.scalar_type(),
                "B must have the same dtype as X");
  }
  if (HAS_Z) {
    TORCH_CHECK(Z.scalar_type() == X.scalar_type(),
                "Z must have the same dtype as X");
  }
  TORCH_CHECK(Rstd.scalar_type() == at::ScalarType::Float,
              "Rstd must be float32");
  if (!IS_RMS_NORM) {
    TORCH_CHECK(Mean.scalar_type() == at::ScalarType::Float,
                "Mean must be float32");
  }

  TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
  if (HAS_BIAS) {
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  }
  TORCH_CHECK(Rstd.is_contiguous(), "Rstd must be contiguous");
  if (!IS_RMS_NORM) {
    TORCH_CHECK(Mean.is_contiguous(), "Mean must be contiguous");
  }

  TORCH_CHECK(stride_x_row > 0, "stride_x_row must be positive");
  TORCH_CHECK(stride_y_row > 0, "stride_y_row must be positive");
  if (HAS_Z) {
    TORCH_CHECK(stride_z_row > 0, "stride_z_row must be positive");
  }

  TORCH_CHECK(W.numel() % N == 0,
              "W.numel() must be divisible by N to infer the group count");
  const int64_t groups = W.numel() / N;
  TORCH_CHECK(groups > 0, "group count inferred from W must be positive");
  TORCH_CHECK(groups <= 65535,
              "group count is too large for CUDA grid.y in this kernel");
  const int64_t row_elems = groups * N;
  if (HAS_BIAS) {
    TORCH_CHECK(B.numel() >= row_elems,
                "B must contain at least groups * N elements");
  }
  TORCH_CHECK(Rstd.numel() >= groups * M,
              "Rstd must contain at least groups * M elements");
  if (!IS_RMS_NORM) {
    TORCH_CHECK(Mean.numel() >= groups * M,
                "Mean must contain at least groups * M elements");
  }

  TORCH_CHECK(stride_x_row >= row_elems,
              "stride_x_row must be at least groups * N");
  TORCH_CHECK(stride_y_row >= row_elems,
              "stride_y_row must be at least groups * N");
  if (HAS_Z) {
    TORCH_CHECK(stride_z_row >= row_elems,
                "stride_z_row must be at least groups * N");
  }

  ln1p_check_strided_capacity(X, M, stride_x_row, row_elems, "X");
  ln1p_check_strided_capacity(Y, M, stride_y_row, row_elems, "Y");
  if (HAS_Z) {
    ln1p_check_strided_capacity(Z, M, stride_z_row, row_elems, "Z");
  }

  const GateActivation activation = ln1p_parse_activation(ACTIVATION);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LN1P_DISPATCH_CASE(HAS_BIAS_V, HAS_Z_V, NORM_BEFORE_GATE_V, RMS_V)    \
  ln1p_dispatch_activation<HAS_BIAS_V, HAS_Z_V, NORM_BEFORE_GATE_V, RMS_V>(   \
      X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M,  \
      N, groups, eps, activation, stream)

  if (HAS_BIAS) {
    if (HAS_Z) {
      if (NORM_BEFORE_GATE) {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(true, true, true, true);
        } else {
          LN1P_DISPATCH_CASE(true, true, true, false);
        }
      } else {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(true, true, false, true);
        } else {
          LN1P_DISPATCH_CASE(true, true, false, false);
        }
      }
    } else {
      if (IS_RMS_NORM) {
        LN1P_DISPATCH_CASE(true, false, false, true);
      } else {
        LN1P_DISPATCH_CASE(true, false, false, false);
      }
    }
  } else {
    if (HAS_Z) {
      if (NORM_BEFORE_GATE) {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(false, true, true, true);
        } else {
          LN1P_DISPATCH_CASE(false, true, true, false);
        }
      } else {
        if (IS_RMS_NORM) {
          LN1P_DISPATCH_CASE(false, true, false, true);
        } else {
          LN1P_DISPATCH_CASE(false, true, false, false);
        }
      }
    } else {
      if (IS_RMS_NORM) {
        LN1P_DISPATCH_CASE(false, false, false, true);
      } else {
        LN1P_DISPATCH_CASE(false, false, false, false);
      }
    }
  }

#undef LN1P_DISPATCH_CASE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at
