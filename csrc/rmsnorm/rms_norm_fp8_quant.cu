// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"

#include <stdint.h>
#include <limits>
#include <stdio.h>
#include <cmath>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <torch/extension.h>

#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "reduce.h"
#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #include <cub/util_type.cuh>
#else
  #include <hipcub/hipcub.hpp>
  #include <hipcub/util_type.hpp>
#endif

namespace at{
namespace native{

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}


// float8_e4m3fn + float8_e5m2
#ifndef USE_ROCM
  #include <c10/util/Float8_e4m3fn.h>
  #include <c10/util/Float8_e5m2.h>
  #define MAYBE_HOST_DEVICE C10_HOST_DEVICE
#else
  #include <ATen/hip/HIPContext.h>
  #include <c10/util/Float8_e4m3fn.h>
  #include <c10/util/Float8_e4m3fnuz.h>
  #include <c10/util/Float8_e5m2.h>
  // ROCm doesn't seem to need C10_HOST_DEVICE for static constexpr
  #define MAYBE_HOST_DEVICE
#endif
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e5m2> ||    
                                      std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

// Using the default max value from pytorch (240.0 0x7F) will cause accuracy
// issues when running dynamic quantization. Here use 224.0 0x7E for rocm.
template <>
struct quant_type_max<c10::Float8_e4m3fnuz> {
  static constexpr c10::Float8_e4m3fnuz val() {
    return c10::Float8_e4m3fnuz(0x7E, c10::Float8_e4m3fnuz::from_bits());
  }
};

template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e5m2> ||    
                                      std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

template <>
struct min_scaling_factor<int8_t> {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return std::numeric_limits<float>::epsilon();
  }
};
// float8_e4m3fn + float8_e5m2

    inline __device__ uint8_t float_to_fp8e5m2(float f) {
    constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
    constexpr uint32_t fp8_max = UINT32_C(143) << 23;
    constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
    } else {
        if (f_bits < (UINT32_C(113) << 23)) {
        f_bits = c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits)
                    + c10::detail::fp32_from_bits(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
        } else {
        uint32_t mant_odd = (f_bits >> 21) & 1;
        f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
        f_bits += mant_odd;
        result = static_cast<uint8_t>(f_bits >> 21);
        }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
    }
  // lj
  // FP32 Constants
  constexpr int fp32_bits = 32;
  constexpr int fp32_mant_bits = 23;
  constexpr int fp32_bias = 127;

  // FP8 E4M3 Constants (通常用于 forward/activation)
  constexpr int e4m3_mant_bits = 3;
  constexpr int e4m3_bias = 7; 

  // FP8 E5M2 Constants (通常用于 backward/gradient)
  constexpr int e5m2_bits = 8;
  inline __device__ uint8_t float_to_fp8e4m3(float src) {
    // conversion from float to unsigned int(32 bits) for convience to fetching each bit
    uint32_t __src = *(unsigned int*)&src;
    // fetch sign bits
    uint8_t sign_bits = (__src & 0x80000000) >> (fp32_bits - e5m2_bits);
    // fetch exponent bitss
    uint32_t exp_bits  = __src & 0x7f800000;
    // fetch mantissa bits
    uint32_t mant_bits = __src & 0x007fffff;
    // fetch absolute value
    uint32_t data_scale = __src & 0x7fffffff;
    // categorical discussions
    /* NAN */
    uint8_t result = 0x0;
    if (exp_bits == 0x7f800000 and mant_bits > 0x0) {
        // result = sign_bits | 0x7f; // output qNAN
        result = 0x7f; // for NV's __nv_cvt_float_to_fp8:cvt.rn.satfinite.e4m3x2.f32, output are all 0x7f
    }
    /* inf or greater than MAX value of E5M2 */
    else if ((exp_bits == 0x7f800000 and mant_bits == 0x0) or (data_scale > 0x43e00000)) {
        result = sign_bits | 0x7e; // output MAX
    }
    /* less than MIN of denorm */
    else if (exp_bits <= 0x3a800000) {
        result = (exp_bits == 0x3a800000 and mant_bits > 0x0) ? sign_bits | 0x1: sign_bits;
    }
    /* others */
    else {
        /* norm fp32 can be represented as denorm fp8 / norm */
        mant_bits = exp_bits < 0x3c800000 ? (0x800000 | mant_bits) >> ((0x3c800000 - exp_bits) >> fp32_mant_bits): mant_bits;
        exp_bits  = exp_bits < 0x3c800000 ? 0x0: ((exp_bits >> fp32_mant_bits) - (fp32_bias - e4m3_bias)) << e4m3_mant_bits;
        // get discard bits
        uint32_t discard_bits = mant_bits & 0xfffff;
        // rounding
        bool carry_a_bit = discard_bits > 0x80000 or (discard_bits == 0x80000 and (mant_bits & 0x100000));
        mant_bits = (mant_bits & 0x700000) >> (fp32_mant_bits - e4m3_mant_bits);
        mant_bits = carry_a_bit ? mant_bits + 1: mant_bits;
        result = sign_bits + exp_bits + mant_bits; // + rather than |, since mant may carry a bit to exp
    }
    return result;
}

    inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c, float d)
    {
    #if defined(__gfx938__) || defined(__gfx946__)
    int32_t res = 0;
    res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
    return static_cast<uint32_t>(res);
    #else
    int32_t out = 0;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d);
    return static_cast<uint32_t>(out);
    #endif
    }


    inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c, float d)
    {
    #if defined(__gfx938__) || defined(__gfx946__)
    int32_t res = 0;
    res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
    return static_cast<uint32_t>(res);
    #else
    int32_t out = 0;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
    return static_cast<uint32_t>(out);
    #endif
    }

namespace rms_fp8_detail {

#if defined(__gfx946__)
using v2f = float __attribute__((ext_vector_type(2)));
using v2s = short __attribute__((ext_vector_type(2)));
using v2h = _Float16 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v2f bf16x2_to_v2f(uint32_t packed) {
  uint32_t lo = (packed & 0xFFFFU) << 16;
  uint32_t hi = packed & 0xFFFF0000U;
  return v2f{__uint_as_float(lo), __uint_as_float(hi)};
}

__device__ __forceinline__ v2f f16x2_to_v2f(uint32_t packed) {
  __half2 h2 = __builtin_bit_cast(__half2, packed);
  float2 f = __half22float2(h2);
  return v2f{f.x, f.y};
}

// VOP3A RNE packed conversion (matches SQ_WAVE_MODE default RNE; never RTZ).
__device__ __forceinline__ uint32_t v2f_to_bf16x2_rn(v2f v) {
  uint32_t res;
  asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(res) : "v"(v[0]), "v"(v[1]));
  return res;
}

__device__ __forceinline__ uint32_t v2f_to_f16x2_rn(v2f v) {
  uint32_t res;
  asm("v_cvt_pk_f16_f32 %0, %1, %2" : "=v"(res) : "v"(v[0]), "v"(v[1]));
  return res;
}

// VOP3P 2xBF16/FP16 dot-product accumulate into FP32 (non-residual Pass 1 only).
__device__ __forceinline__ float dot2_sq_bf16(uint32_t packed_bf16, float acc) {
  asm("v_dot2_f32_bf16 %0, %1, %1, %0" : "+v"(acc) : "v"(packed_bf16));
  return acc;
}

__device__ __forceinline__ float dot2_sq_f16(uint32_t packed_f16, float acc) {
#if defined(__has_builtin) && __has_builtin(__builtin_hcu_fdot2)
  v2h h = __builtin_bit_cast(v2h, packed_f16);
  return __builtin_hcu_fdot2(h, h, acc, false);
#else
  asm("v_dot2_f32_f16 %0, %1, %1, %0" : "+v"(acc) : "v"(packed_f16));
  return acc;
#endif
}

// Packed BF16/FP16 absmax (Group 1 DTK 27.04-1010/1114 verified errata R2/R3).
__device__ __forceinline__ uint32_t pk_absmax_bf16_u32(uint32_t a_u32, uint32_t b_u32) {
  v2s va = __builtin_bit_cast(v2s, a_u32 & 0x7FFF7FFFU);
  v2s vb = __builtin_bit_cast(v2s, b_u32 & 0x7FFF7FFFU);
  return __builtin_bit_cast(uint32_t, __builtin_hcu_pk_max_bf16(va, vb, 0, 7));
}

__device__ __forceinline__ float bf16_pair_max_to_f32(uint32_t pair_abs_u32) {
  uint32_t lo = (pair_abs_u32 & 0xFFFFU) << 16;
  uint32_t hi = pair_abs_u32 & 0xFFFF0000U;
  return fmaxf(__uint_as_float(lo), __uint_as_float(hi));
}

__device__ __forceinline__ uint32_t pk_absmax_f16_u32(uint32_t a_u32, uint32_t b_u32) {
  uint32_t a_abs = a_u32 & 0x7FFF7FFFU;
  uint32_t b_abs = b_u32 & 0x7FFF7FFFU;
  uint32_t res;
  asm("v_pk_max_f16 %0, %1, %2" : "=v"(res) : "v"(a_abs), "v"(b_abs));
  return res;
}

__device__ __forceinline__ float f16_pair_max_to_f32(uint32_t pair_abs_u32) {
  __half2 h2 = __builtin_bit_cast(__half2, pair_abs_u32);
  float2 f = __half22float2(h2);
  return fmaxf(f.x, f.y);
}

// 1-Pass VALU cross-half-wave swap (Group 1 verified dual "+v" asm, errata R4).
__device__ __forceinline__ float permlane16_swap_fadd(float val) {
  float other = val;
  asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return val + other;
}

__device__ __forceinline__ float permlane32_swap_fadd(float val) {
  float other = val;
  asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return val + other;
}

__device__ __forceinline__ float permlane16_swap_fmax(float val) {
  float other = val;
  asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return fmaxf(val, other);
}

__device__ __forceinline__ float permlane32_swap_fmax(float val) {
  float other = val;
  asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return fmaxf(val, other);
}

// Route A (arbitrary FP32 scale): v2f lowers to v_pk_mul_f32 + hardware cvt_pk_{fp8,bf8}_f32.
template <bool IS_E4M3>
__device__ __forceinline__ uint32_t f32x4_mul_to_fp8x4(v2f v01, v2f v23, float inv_s) {
  const v2f vs = {inv_s, inv_s};
  v01 = v01 * vs;
  v23 = v23 * vs;
  if constexpr (IS_E4M3) {
    return float_to_fp8e4m3_x4(v01[0], v01[1], v23[0], v23[1]);
  } else {
    return float_to_fp8e5m2_x4(v01[0], v01[1], v23[0], v23[1]);
  }
}
#endif  // defined(__gfx946__)

template <typename T, int reducesize = C10_WARP_SIZE>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#if defined(__gfx946__)
  if constexpr (std::is_same_v<T, float> && reducesize == 64) {
    val = permlane32_swap_fadd(val);
    val = permlane16_swap_fadd(val);
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      val += __shfl_down(val, offset);
    }
    return val;
  }
#endif
#pragma unroll
  for (int offset = reducesize / 2; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T, int block_size = 512>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  val = warp_reduce_sum<T, C10_WARP_SIZE>(val);
  if constexpr (block_size == C10_WARP_SIZE) {
    return val;
  } else {
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0 && wid < share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0 && lid < share_size) {
      val = warp_reduce_sum<T, share_size>(shared[lid]);
    }
    return val;
  }
}

template <typename T, int reducesize = C10_WARP_SIZE>
__device__ __forceinline__ T warp_reduce_max(T val) {
#if defined(__gfx946__)
  if constexpr (std::is_same_v<T, float> && reducesize == 64) {
    val = permlane32_swap_fmax(val);
    val = permlane16_swap_fmax(val);
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
  }
#endif
#pragma unroll
  for (int offset = reducesize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, WARP_SHFL_DOWN(val, offset));
  }
  return val;
}

template <typename T, int block_size = 512>
__device__ __forceinline__ T block_reduce_max(T val, T* shared) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  val = warp_reduce_max<T, C10_WARP_SIZE>(val);
  if constexpr (block_size == C10_WARP_SIZE) {
    return val;
  } else {
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0 && wid < share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0 && lid < share_size) {
      val = warp_reduce_max<T, share_size>(shared[lid]);
    }
    return val;
  }
}

template <int block_size, typename T>
__device__ __forceinline__ T broadcast_lane0(T val, T* shared_slot) {
  if constexpr (block_size == C10_WARP_SIZE && sizeof(T) == 4) {
#ifdef USE_ROCM
    return __shfl(val, 0);
#else
    return __shfl_sync(0xffffffff, val, 0);
#endif
  } else {
    if (threadIdx.x == 0) {
      *shared_slot = val;
    }
    __syncthreads();
    return *shared_slot;
  }
}

template <bool IS_E4M3>
__device__ __forceinline__ uint8_t scalar_float_to_fp8(float x) {
  if constexpr (IS_E4M3) {
    return float_to_fp8e4m3(x);
  } else {
    return float_to_fp8e5m2(x);
  }
}

}  // namespace rms_fp8_detail

template <typename scalar_t, typename scalar_out_t, typename T_ACC,
          int Vec = 4, int block_size = 512,
          bool inpit_update = false, bool IS_GEMMA = false, bool IS_E4M3 = true>
__global__ void rms_norm_fp8_quant_(
    scalar_t* input,
    scalar_t* residual,
    scalar_t* gamma,
    int32_t const cols,
    float const eps,
    scalar_out_t* out,
    float* scales,
    int64_t stride,
    int num_tokens) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale;

  constexpr float fp8type_max_v = IS_E4M3 ? 448.0f : 57344.0f;
  constexpr float inv_fp8_max_v = 1.0f / fp8type_max_v;
  constexpr float min_scaling = 1.0f / (fp8type_max_v * 512.0f);

  const int tid = threadIdx.x;
  const int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;

  for (int bid = blockIdx.x; bid < num_tokens; bid += gridDim.x) {
    const int64_t out_idx = (static_cast<int64_t>(bid) * tcol + tid) * Vec;
    const int64_t in_idx = static_cast<int64_t>(bid) * stride + static_cast<int64_t>(tid) * Vec;

#if defined(__gfx946__)
    if constexpr (Vec == 16 && sizeof(scalar_out_t) == 1 &&
                  (std::is_same_v<scalar_t, at::BFloat16> || std::is_same_v<scalar_t, at::Half>)) {
      constexpr bool IS_BF16 = std::is_same_v<scalar_t, at::BFloat16>;
      // Reuse raw_vec[8] (8 VGPRs) in-place as elem_pk[8] across barriers to fit 512 KB VGPR/CU.
      uint32_t raw_vec[8];
      float val = 0.0f;
      if (tid < tcol) {
        *(LoadT*)raw_vec = *(const LoadT*)(input + in_idx);
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          if constexpr (IS_BF16) {
            val = rms_fp8_detail::dot2_sq_bf16(raw_vec[p], val);
          } else {
            val = rms_fp8_detail::dot2_sq_f16(raw_vec[p], val);
          }
        }
      }
      val = rms_fp8_detail::block_reduce_sum<float, block_size>(val, val_shared);
      float trstd = 0.0f;
      if (tid == 0) {
        trstd = c10::cuda::compat::rsqrt(val / static_cast<float>(cols) + eps);
      }
      trstd = rms_fp8_detail::broadcast_lane0<block_size>(trstd, &s_rstd);

      float block_absmax = 0.0f;
      if (tid < tcol) {
        uint32_t w_raw[8];
        *(LoadT*)w_raw = *(const LoadT*)(gamma + static_cast<int64_t>(tid) * Vec);
        const rms_fp8_detail::v2f v_rstd = {trstd, trstd};
        const rms_fp8_detail::v2f v_one = {1.0f, 1.0f};
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          rms_fp8_detail::v2f x_f = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(raw_vec[p])
                                            : rms_fp8_detail::f16x2_to_v2f(raw_vec[p]);
          rms_fp8_detail::v2f w_f = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(w_raw[p])
                                            : rms_fp8_detail::f16x2_to_v2f(w_raw[p]);
          if constexpr (IS_GEMMA) {
            w_f = w_f + v_one;
          }
          rms_fp8_detail::v2f y_f = (x_f * v_rstd) * w_f;
          raw_vec[p] = IS_BF16 ? rms_fp8_detail::v2f_to_bf16x2_rn(y_f)
                               : rms_fp8_detail::v2f_to_f16x2_rn(y_f);
        }
        uint32_t m0, m1, m2, m3, m01, m23, m_pair;
        if constexpr (IS_BF16) {
          m0 = rms_fp8_detail::pk_absmax_bf16_u32(raw_vec[0], raw_vec[1]);
          m1 = rms_fp8_detail::pk_absmax_bf16_u32(raw_vec[2], raw_vec[3]);
          m2 = rms_fp8_detail::pk_absmax_bf16_u32(raw_vec[4], raw_vec[5]);
          m3 = rms_fp8_detail::pk_absmax_bf16_u32(raw_vec[6], raw_vec[7]);
          m01 = rms_fp8_detail::pk_absmax_bf16_u32(m0, m1);
          m23 = rms_fp8_detail::pk_absmax_bf16_u32(m2, m3);
          m_pair = rms_fp8_detail::pk_absmax_bf16_u32(m01, m23);
          block_absmax = rms_fp8_detail::bf16_pair_max_to_f32(m_pair);
        } else {
          m0 = rms_fp8_detail::pk_absmax_f16_u32(raw_vec[0], raw_vec[1]);
          m1 = rms_fp8_detail::pk_absmax_f16_u32(raw_vec[2], raw_vec[3]);
          m2 = rms_fp8_detail::pk_absmax_f16_u32(raw_vec[4], raw_vec[5]);
          m3 = rms_fp8_detail::pk_absmax_f16_u32(raw_vec[6], raw_vec[7]);
          m01 = rms_fp8_detail::pk_absmax_f16_u32(m0, m1);
          m23 = rms_fp8_detail::pk_absmax_f16_u32(m2, m3);
          m_pair = rms_fp8_detail::pk_absmax_f16_u32(m01, m23);
          block_absmax = rms_fp8_detail::f16_pair_max_to_f32(m_pair);
        }
      }
      block_absmax = rms_fp8_detail::block_reduce_max<float, block_size>(block_absmax, val_shared);
      float token_scale = 0.0f;
      if (tid == 0) {
        token_scale = fmaxf(block_absmax * inv_fp8_max_v, min_scaling);
        scales[bid] = token_scale;
      }
      token_scale = rms_fp8_detail::broadcast_lane0<block_size>(token_scale, &s_token_scale);
      const float inv_s = 1.0f / token_scale;

      if (tid < tcol) {
        scalar_out_t out_vec[Vec];
        uint32_t* out_u32 = reinterpret_cast<uint32_t*>(out_vec);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          rms_fp8_detail::v2f v01 = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(raw_vec[2 * p])
                                            : rms_fp8_detail::f16x2_to_v2f(raw_vec[2 * p]);
          rms_fp8_detail::v2f v23 = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(raw_vec[2 * p + 1])
                                            : rms_fp8_detail::f16x2_to_v2f(raw_vec[2 * p + 1]);
          out_u32[p] = rms_fp8_detail::f32x4_mul_to_fp8x4<IS_E4M3>(v01, v23, inv_s);
        }
        *(LoadT_out*)(out + out_idx) = *(LoadT_out*)out_vec;
        if constexpr (inpit_update) {
          *(LoadT*)(input + in_idx) = *(LoadT*)raw_vec;
        }
      }
      continue;
    }
#endif  // defined(__gfx946__)

    scalar_t intput_vec[Vec];
    T_ACC val = 0;
    if (tid < tcol) {
      *(LoadT*)intput_vec = *(const LoadT*)(input + in_idx);
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        const T_ACC x = static_cast<T_ACC>(intput_vec[ii]);
        val += x * x;
      }
    }
    val = rms_fp8_detail::block_reduce_sum<T_ACC, block_size>(val, val_shared);
    T_ACC trstd = 0;
    if (tid == 0) {
      trstd = c10::cuda::compat::rsqrt(val / static_cast<T_ACC>(cols) + static_cast<T_ACC>(eps));
    }
    trstd = rms_fp8_detail::broadcast_lane0<block_size>(trstd, &s_rstd);

    float block_absmax_val_maybe = 0.f;
    scalar_t elem[Vec];
    if (tid < tcol) {
      scalar_t weight_vec[Vec];
      *(LoadT*)weight_vec = *(const LoadT*)(gamma + static_cast<int64_t>(tid) * Vec);
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        T_ACC w_val = static_cast<T_ACC>(weight_vec[ii]);
        if constexpr (IS_GEMMA) {
          w_val += static_cast<T_ACC>(1.0f);
        }
        elem[ii] = static_cast<scalar_t>(static_cast<T_ACC>(intput_vec[ii]) * trstd * w_val);
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(static_cast<float>(elem[ii])));
      }
    }
    block_absmax_val_maybe = rms_fp8_detail::block_reduce_max<float, block_size>(
        block_absmax_val_maybe, reinterpret_cast<float*>(val_shared));

    float token_scale = 0.0f;
    if (tid == 0) {
      token_scale = fmaxf(block_absmax_val_maybe * inv_fp8_max_v, min_scaling);
      scales[bid] = token_scale;
    }
    token_scale = rms_fp8_detail::broadcast_lane0<block_size>(token_scale, &s_token_scale);
    const float inv_s = 1.0f / token_scale;

    scalar_out_t out_vec[Vec];
    constexpr int loop = Vec / 4;
    if (tid < tcol) {
#pragma unroll
      for (int ii = 0; ii < loop; ii++) {
        const float e0 = static_cast<float>(elem[ii * 4 + 0]);
        const float e1 = static_cast<float>(elem[ii * 4 + 1]);
        const float e2 = static_cast<float>(elem[ii * 4 + 2]);
        const float e3 = static_cast<float>(elem[ii * 4 + 3]);
#if defined(__gfx946__)
        reinterpret_cast<uint32_t*>(out_vec)[ii] =
            rms_fp8_detail::f32x4_mul_to_fp8x4<IS_E4M3>(
                rms_fp8_detail::v2f{e0, e1}, rms_fp8_detail::v2f{e2, e3}, inv_s);
#else
        if constexpr (IS_E4M3) {
          reinterpret_cast<uint32_t*>(out_vec)[ii] =
              float_to_fp8e4m3_x4(e0 * inv_s, e1 * inv_s, e2 * inv_s, e3 * inv_s);
        } else {
          reinterpret_cast<uint32_t*>(out_vec)[ii] =
              float_to_fp8e5m2_x4(e0 * inv_s, e1 * inv_s, e2 * inv_s, e3 * inv_s);
        }
#endif
      }
      *(LoadT_out*)(out + out_idx) = *(LoadT_out*)out_vec;
      if constexpr (inpit_update) {
        *(LoadT*)(input + in_idx) = *(LoadT*)elem;
      }
    }
  }
}

template <typename scalar_t, typename scalar_out_t, typename T_ACC,
          int Vec, int block_size,
          bool inpit_update = false, bool IS_GEMMA = false, bool IS_E4M3 = true>
__global__ void rms_norm_fp8_quant_loop(
    scalar_t* input,
    scalar_t* residual,
    scalar_t* gamma,
    int32_t const cols,
    float const eps,
    scalar_out_t* out,
    float* scales,
    int64_t stride,
    int64_t out_stride) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale;

  constexpr float fp8type_max_v = IS_E4M3 ? 448.0f : 57344.0f;
  constexpr float inv_fp8_max_v = 1.0f / fp8type_max_v;
  constexpr float min_scaling = 1.0f / (fp8type_max_v * 512.0f);

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t in_rows = static_cast<int64_t>(bid) * stride;
  const int64_t out_rows = static_cast<int64_t>(bid) * out_stride;

  T_ACC val = 0;
  for (int i = tid; i < cols; i += block_size) {
    const T_ACC x = static_cast<T_ACC>(input[in_rows + i]);
    val += x * x;
  }
  val = rms_fp8_detail::block_reduce_sum<T_ACC, block_size>(val, val_shared);
  T_ACC trstd = 0;
  if (tid == 0) {
    trstd = c10::cuda::compat::rsqrt(val / static_cast<T_ACC>(cols) + static_cast<T_ACC>(eps));
  }
  trstd = rms_fp8_detail::broadcast_lane0<block_size>(trstd, &s_rstd);

  float block_absmax_val_maybe = 0.0f;
  for (int i = tid; i < cols; i += block_size) {
    T_ACC w_val = static_cast<T_ACC>(gamma[i]);
    if constexpr (IS_GEMMA) {
      w_val += static_cast<T_ACC>(1.0f);
    }
    const scalar_t elem = static_cast<scalar_t>(static_cast<T_ACC>(input[in_rows + i]) * trstd * w_val);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(static_cast<float>(elem)));
  }
  block_absmax_val_maybe = rms_fp8_detail::block_reduce_max<float, block_size>(
      block_absmax_val_maybe, reinterpret_cast<float*>(val_shared));

  float token_scale = 0.0f;
  if (tid == 0) {
    token_scale = fmaxf(block_absmax_val_maybe * inv_fp8_max_v, min_scaling);
    scales[bid] = token_scale;
  }
  token_scale = rms_fp8_detail::broadcast_lane0<block_size>(token_scale, &s_token_scale);
  const float inv_s = 1.0f / token_scale;

  for (int i = tid; i < cols; i += block_size) {
    T_ACC w_val = static_cast<T_ACC>(gamma[i]);
    if constexpr (IS_GEMMA) {
      w_val += static_cast<T_ACC>(1.0f);
    }
    const scalar_t elem = static_cast<scalar_t>(static_cast<T_ACC>(input[in_rows + i]) * trstd * w_val);
    out[out_rows + i] = static_cast<scalar_out_t>(
        rms_fp8_detail::scalar_float_to_fp8<IS_E4M3>(static_cast<float>(elem) * inv_s));
    if constexpr (inpit_update) {
      input[in_rows + i] = elem;
    }
  }
}


template <typename scalar_t, typename scalar_out_t, typename T_ACC,
          int Vec = 4, int block_size = 512,
          bool inpit_update = false, bool IS_GEMMA = false, bool IS_E4M3 = true>
__global__ void rms_norm_fp8_quant_residual(
    scalar_t* input,
    scalar_t* residual,
    scalar_t* gamma,
    int cols,
    float const eps,
    scalar_out_t* out,
    float* scales,
    int64_t stride,
    int num_tokens) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale;

  constexpr float fp8type_max_v = IS_E4M3 ? 448.0f : 57344.0f;
  constexpr float inv_fp8_max_v = 1.0f / fp8type_max_v;
  constexpr float min_scaling = 1.0f / (fp8type_max_v * 512.0f);

  const int tid = threadIdx.x;
  const int tcol = cols / Vec;
  using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
  using LoadT_out = at::native::memory::aligned_vector<scalar_out_t, Vec>;

  for (int bid = blockIdx.x; bid < num_tokens; bid += gridDim.x) {
    const int64_t idx = (static_cast<int64_t>(bid) * tcol + tid) * Vec;

#if defined(__gfx946__)
    if constexpr (Vec == 16 && sizeof(scalar_out_t) == 1 &&
                  (std::is_same_v<scalar_t, at::BFloat16> || std::is_same_v<scalar_t, at::Half>)) {
      constexpr bool IS_BF16 = std::is_same_v<scalar_t, at::BFloat16>;
      // Aligned to FP32 residual sum contract: keep summed_v2[8] in FP32 through norm;
      // overlap residual store with Reduction #1 and shrink live state to elem_pk[8] across Reduction #2.
      rms_fp8_detail::v2f summed_v2[8];
      float val = 0.0f;
      if (tid < tcol) {
        uint32_t raw_in[8];
        uint32_t raw_res[8];
        *(LoadT*)raw_in = *(const LoadT*)(input + idx);
        *(LoadT*)raw_res = *(const LoadT*)(residual + idx);
        rms_fp8_detail::v2f var_acc = {0.0f, 0.0f};
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          rms_fp8_detail::v2f x_f = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(raw_in[p])
                                            : rms_fp8_detail::f16x2_to_v2f(raw_in[p]);
          rms_fp8_detail::v2f r_f = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(raw_res[p])
                                            : rms_fp8_detail::f16x2_to_v2f(raw_res[p]);
          rms_fp8_detail::v2f s = x_f + r_f;      // v_pk_add_f32
          summed_v2[p] = s;
          var_acc = var_acc + s * s;              // v_pk_fma_f32
          raw_res[p] = IS_BF16 ? rms_fp8_detail::v2f_to_bf16x2_rn(s)
                               : rms_fp8_detail::v2f_to_f16x2_rn(s);
        }
        val = var_acc[0] + var_acc[1];
        *(LoadT*)(residual + idx) = *(LoadT*)raw_res;
      }

      val = rms_fp8_detail::block_reduce_sum<float, block_size>(val, val_shared);
      float trstd = 0.0f;
      if (tid == 0) {
        trstd = c10::cuda::compat::rsqrt(val / static_cast<float>(cols) + eps);
      }
      trstd = rms_fp8_detail::broadcast_lane0<block_size>(trstd, &s_rstd);

      uint32_t elem_pk[8];
      float block_absmax = 0.0f;
      if (tid < tcol) {
        uint32_t w_raw[8];
        *(LoadT*)w_raw = *(const LoadT*)(gamma + static_cast<int64_t>(tid) * Vec);
        const rms_fp8_detail::v2f v_rstd = {trstd, trstd};
        const rms_fp8_detail::v2f v_one = {1.0f, 1.0f};
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          rms_fp8_detail::v2f w_f = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(w_raw[p])
                                            : rms_fp8_detail::f16x2_to_v2f(w_raw[p]);
          if constexpr (IS_GEMMA) {
            w_f = w_f + v_one;
          }
          rms_fp8_detail::v2f y_f = (summed_v2[p] * v_rstd) * w_f;
          elem_pk[p] = IS_BF16 ? rms_fp8_detail::v2f_to_bf16x2_rn(y_f)
                               : rms_fp8_detail::v2f_to_f16x2_rn(y_f);
        }
        uint32_t m0, m1, m2, m3, m01, m23, m_pair;
        if constexpr (IS_BF16) {
          m0 = rms_fp8_detail::pk_absmax_bf16_u32(elem_pk[0], elem_pk[1]);
          m1 = rms_fp8_detail::pk_absmax_bf16_u32(elem_pk[2], elem_pk[3]);
          m2 = rms_fp8_detail::pk_absmax_bf16_u32(elem_pk[4], elem_pk[5]);
          m3 = rms_fp8_detail::pk_absmax_bf16_u32(elem_pk[6], elem_pk[7]);
          m01 = rms_fp8_detail::pk_absmax_bf16_u32(m0, m1);
          m23 = rms_fp8_detail::pk_absmax_bf16_u32(m2, m3);
          m_pair = rms_fp8_detail::pk_absmax_bf16_u32(m01, m23);
          block_absmax = rms_fp8_detail::bf16_pair_max_to_f32(m_pair);
        } else {
          m0 = rms_fp8_detail::pk_absmax_f16_u32(elem_pk[0], elem_pk[1]);
          m1 = rms_fp8_detail::pk_absmax_f16_u32(elem_pk[2], elem_pk[3]);
          m2 = rms_fp8_detail::pk_absmax_f16_u32(elem_pk[4], elem_pk[5]);
          m3 = rms_fp8_detail::pk_absmax_f16_u32(elem_pk[6], elem_pk[7]);
          m01 = rms_fp8_detail::pk_absmax_f16_u32(m0, m1);
          m23 = rms_fp8_detail::pk_absmax_f16_u32(m2, m3);
          m_pair = rms_fp8_detail::pk_absmax_f16_u32(m01, m23);
          block_absmax = rms_fp8_detail::f16_pair_max_to_f32(m_pair);
        }
      }

      block_absmax = rms_fp8_detail::block_reduce_max<float, block_size>(block_absmax, val_shared);
      float token_scale = 0.0f;
      if (tid == 0) {
        token_scale = fmaxf(block_absmax * inv_fp8_max_v, min_scaling);
        scales[bid] = token_scale;
      }
      token_scale = rms_fp8_detail::broadcast_lane0<block_size>(token_scale, &s_token_scale);
      const float inv_s = 1.0f / token_scale;

      if (tid < tcol) {
        scalar_out_t out_vec[Vec];
        uint32_t* out_u32 = reinterpret_cast<uint32_t*>(out_vec);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          rms_fp8_detail::v2f v01 = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(elem_pk[2 * p])
                                            : rms_fp8_detail::f16x2_to_v2f(elem_pk[2 * p]);
          rms_fp8_detail::v2f v23 = IS_BF16 ? rms_fp8_detail::bf16x2_to_v2f(elem_pk[2 * p + 1])
                                            : rms_fp8_detail::f16x2_to_v2f(elem_pk[2 * p + 1]);
          out_u32[p] = rms_fp8_detail::f32x4_mul_to_fp8x4<IS_E4M3>(v01, v23, inv_s);
        }
        *(LoadT_out*)(out + idx) = *(LoadT_out*)out_vec;
        if constexpr (inpit_update) {
          *(LoadT*)(input + idx) = *(LoadT*)elem_pk;
        }
      }
      continue;
    }
#endif  // defined(__gfx946__)

    // Aligned FP32 residual sum path for non-gfx946 and FP32/FP64 dtypes.
    T_ACC summed_vec[Vec];
    T_ACC val = 0;
    if (tid < tcol) {
      scalar_t intput_vec[Vec];
      scalar_t residual_vec[Vec];
      *(LoadT*)intput_vec = *(const LoadT*)(input + idx);
      *(LoadT*)residual_vec = *(const LoadT*)(residual + idx);
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        const T_ACC s = static_cast<T_ACC>(intput_vec[ii]) + static_cast<T_ACC>(residual_vec[ii]);
        summed_vec[ii] = s;
        val += s * s;
        residual_vec[ii] = static_cast<scalar_t>(s);
      }
      *(LoadT*)(residual + idx) = *(LoadT*)residual_vec;
    }

    val = rms_fp8_detail::block_reduce_sum<T_ACC, block_size>(val, val_shared);
    T_ACC trstd = 0;
    if (tid == 0) {
      trstd = c10::cuda::compat::rsqrt(val / static_cast<T_ACC>(cols) + static_cast<T_ACC>(eps));
    }
    trstd = rms_fp8_detail::broadcast_lane0<block_size>(trstd, &s_rstd);

    float block_absmax_val_maybe = 0.f;
    scalar_t elem[Vec];
    if (tid < tcol) {
      scalar_t weight_vec[Vec];
      *(LoadT*)weight_vec = *(const LoadT*)(gamma + static_cast<int64_t>(tid) * Vec);
#pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        T_ACC w_val = static_cast<T_ACC>(weight_vec[ii]);
        if constexpr (IS_GEMMA) {
          w_val += static_cast<T_ACC>(1.0f);
        }
        elem[ii] = static_cast<scalar_t>(summed_vec[ii] * trstd * w_val);
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(static_cast<float>(elem[ii])));
      }
    }
    block_absmax_val_maybe = rms_fp8_detail::block_reduce_max<float, block_size>(
        block_absmax_val_maybe, reinterpret_cast<float*>(val_shared));

    float token_scale = 0.0f;
    if (tid == 0) {
      token_scale = fmaxf(block_absmax_val_maybe * inv_fp8_max_v, min_scaling);
      scales[bid] = token_scale;
    }
    token_scale = rms_fp8_detail::broadcast_lane0<block_size>(token_scale, &s_token_scale);
    const float inv_s = 1.0f / token_scale;

    scalar_out_t out_vec[Vec];
    constexpr int loop = Vec / 4;
    if (tid < tcol) {
#pragma unroll
      for (int ii = 0; ii < loop; ii++) {
        const float e0 = static_cast<float>(elem[ii * 4 + 0]);
        const float e1 = static_cast<float>(elem[ii * 4 + 1]);
        const float e2 = static_cast<float>(elem[ii * 4 + 2]);
        const float e3 = static_cast<float>(elem[ii * 4 + 3]);
#if defined(__gfx946__)
        reinterpret_cast<uint32_t*>(out_vec)[ii] =
            rms_fp8_detail::f32x4_mul_to_fp8x4<IS_E4M3>(
                rms_fp8_detail::v2f{e0, e1}, rms_fp8_detail::v2f{e2, e3}, inv_s);
#else
        if constexpr (IS_E4M3) {
          reinterpret_cast<uint32_t*>(out_vec)[ii] =
              float_to_fp8e4m3_x4(e0 * inv_s, e1 * inv_s, e2 * inv_s, e3 * inv_s);
        } else {
          reinterpret_cast<uint32_t*>(out_vec)[ii] =
              float_to_fp8e5m2_x4(e0 * inv_s, e1 * inv_s, e2 * inv_s, e3 * inv_s);
        }
#endif
      }
      *(LoadT_out*)(out + idx) = *(LoadT_out*)out_vec;
      if constexpr (inpit_update) {
        *(LoadT*)(input + idx) = *(LoadT*)elem;
      }
    }
  }
}


template <typename scalar_t, typename scalar_out_t, typename T_ACC,
          int Vec = 1, int block_size = 512,
          bool inpit_update = false, bool IS_GEMMA = false, bool IS_E4M3 = true>
__global__ void rms_norm_fp8_quant_residual_loop(
    scalar_t* input,
    scalar_t* residual,
    scalar_t* gamma,
    int cols,
    float const eps,
    scalar_out_t* out,
    float* scales,
    int64_t stride,
    int64_t res_stride,
    int64_t out_stride) {
  constexpr int share_size = block_size / C10_WARP_SIZE;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  __shared__ float s_token_scale;

  constexpr float fp8type_max_v = IS_E4M3 ? 448.0f : 57344.0f;
  constexpr float inv_fp8_max_v = 1.0f / fp8type_max_v;
  constexpr float min_scaling = 1.0f / (fp8type_max_v * 512.0f);

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t in_rows = static_cast<int64_t>(bid) * stride;
  const int64_t res_rows = static_cast<int64_t>(bid) * res_stride;
  const int64_t out_rows = static_cast<int64_t>(bid) * out_stride;

  T_ACC val = 0;
  for (int i = tid; i < cols; i += block_size) {
    const T_ACC x = static_cast<T_ACC>(residual[res_rows + i]) + static_cast<T_ACC>(input[in_rows + i]);
    val += x * x;
  }

  val = rms_fp8_detail::block_reduce_sum<T_ACC, block_size>(val, val_shared);
  T_ACC trstd = 0;
  if (tid == 0) {
    trstd = c10::cuda::compat::rsqrt(val / static_cast<T_ACC>(cols) + static_cast<T_ACC>(eps));
  }
  trstd = rms_fp8_detail::broadcast_lane0<block_size>(trstd, &s_rstd);

  float block_absmax_val_maybe = 0.0f;
  for (int i = tid; i < cols; i += block_size) {
    const T_ACC sum = static_cast<T_ACC>(residual[res_rows + i]) + static_cast<T_ACC>(input[in_rows + i]);
    T_ACC w_val = static_cast<T_ACC>(gamma[i]);
    if constexpr (IS_GEMMA) {
      w_val += static_cast<T_ACC>(1.0f);
    }
    const scalar_t elem = static_cast<scalar_t>(sum * trstd * w_val);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(static_cast<float>(elem)));
  }

  block_absmax_val_maybe = rms_fp8_detail::block_reduce_max<float, block_size>(
      block_absmax_val_maybe, reinterpret_cast<float*>(val_shared));

  float token_scale = 0.0f;
  if (tid == 0) {
    token_scale = fmaxf(block_absmax_val_maybe * inv_fp8_max_v, min_scaling);
    scales[bid] = token_scale;
  }
  token_scale = rms_fp8_detail::broadcast_lane0<block_size>(token_scale, &s_token_scale);
  const float inv_s = 1.0f / token_scale;

  for (int i = tid; i < cols; i += block_size) {
    const T_ACC sum = static_cast<T_ACC>(residual[res_rows + i]) + static_cast<T_ACC>(input[in_rows + i]);
    T_ACC w_val = static_cast<T_ACC>(gamma[i]);
    if constexpr (IS_GEMMA) {
      w_val += static_cast<T_ACC>(1.0f);
    }
    const scalar_t elem = static_cast<scalar_t>(sum * trstd * w_val);
    residual[res_rows + i] = static_cast<scalar_t>(sum);
    out[out_rows + i] = static_cast<scalar_out_t>(
        rms_fp8_detail::scalar_float_to_fp8<IS_E4M3>(static_cast<float>(elem) * inv_s));
    if constexpr (inpit_update) {
      input[in_rows + i] = elem;
    }
  }
}


template <typename scalar_in_t, bool IS_GEMMA>
void fuse_rms_norm_fp8_quant(
  torch::Tensor& out,           // [..., hidden_size]
  torch::Tensor& input,         // [..., hidden_size]
  torch::Tensor const& weight,  // [hidden_size]
  torch::Tensor& scales,        // [num_tokens]
  double const epsilon,         // Variance epsilon used in norm calculation
  std::optional<at::Tensor> residual,
  bool input_flag,
  int fp8type) {
  const int hidden_size = input.size(-1);
  TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
  TORCH_CHECK(
      out.scalar_type() == at::kByte || out.scalar_type() == at::kChar,
      "rms_norm_fp8_quant expects an 8-bit integer output view (uint8/int8)");
  const int num_tokens = static_cast<int>(input.numel() / hidden_size);
  if (num_tokens == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  const auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  const auto out_ptr = reinterpret_cast<std::uintptr_t>(out.data_ptr());
  const int input_ndim = input.dim();
  const int64_t stride = (input_ndim > 1) ? input.stride(-2) : hidden_size;
  const int64_t out_stride = (out.dim() > 1) ? out.stride(-2) : hidden_size;
  const bool input_layout_ok =
      (input.stride(-1) == 1) && ((input_ndim <= 2) || input.is_contiguous());

  scalar_in_t* self_data = input.data_ptr<scalar_in_t>();
  scalar_in_t* other_data = nullptr;
  int64_t res_stride = hidden_size;
  bool residual_flag = false;
  bool res_ok_for_vec = true;
  if (residual.has_value()) {
    residual_flag = true;
    other_data = residual->data_ptr<scalar_in_t>();
    res_stride = (residual->dim() > 1) ? residual->stride(-2) : hidden_size;
    const auto res_ptr = reinterpret_cast<std::uintptr_t>(other_data);
    res_ok_for_vec = (res_ptr % 16 == 0) && input.is_contiguous() && residual->is_contiguous();
  }
  const bool stride_aligned =
      (num_tokens <= 1) ||
      ((static_cast<std::size_t>(stride) * sizeof(scalar_in_t)) % 16 == 0);
  const bool ptrs_are_aligned =
      (inp_ptr % 16 == 0) && (wt_ptr % 16 == 0) && (out_ptr % 16 == 0) &&
      input_layout_ok && out.is_contiguous() && weight.is_contiguous() &&
      stride_aligned && res_ok_for_vec;

  int grid_opt = num_tokens;
  if (num_tokens >= 137 && num_tokens <= 144 &&
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount <= 128) {
    grid_opt = 128;
  }

  using T_ACC = at::acc_type<scalar_in_t, true>;
  scalar_in_t* weight_data = weight.data_ptr<scalar_in_t>();
  float* scaless = scales.data_ptr<float>();
  const float eps_f32 = static_cast<float>(epsilon);

  auto dispatch_fp8_and_update = [&](auto&& launcher) {
    if (fp8type == 0) {
      if (input_flag) {
        launcher(std::true_type{}, std::true_type{});
      } else {
        launcher(std::true_type{}, std::false_type{});
      }
    } else {
      if (input_flag) {
        launcher(std::false_type{}, std::true_type{});
      } else {
        launcher(std::false_type{}, std::false_type{});
      }
    }
  };

  auto dispatch_out_byte = [&](auto&& body) {
    if (out.scalar_type() == at::kByte) {
      body(static_cast<uint8_t*>(out.data_ptr<uint8_t>()));
    } else {
      body(static_cast<int8_t*>(out.data_ptr<int8_t>()));
    }
  };

  if (hidden_size % 16 == 0 && hidden_size <= 16384 && ptrs_are_aligned) {
    dispatch_out_byte([&](auto* out_data) {
      using scalar_out_t = std::remove_pointer_t<decltype(out_data)>;
      constexpr int Vec = 16;

      dispatch_fp8_and_update([&](auto is_e4m3_tag, auto update_tag) {
        constexpr bool IS_E4M3 = decltype(is_e4m3_tag)::value;
        constexpr bool update_input = decltype(update_tag)::value;

        if (residual_flag) {
          if (hidden_size <= 1024) {
            rms_norm_fp8_quant_residual<scalar_in_t, scalar_out_t, T_ACC, Vec, 64, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 64, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else if (hidden_size <= 2048) {
            rms_norm_fp8_quant_residual<scalar_in_t, scalar_out_t, T_ACC, Vec, 128, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 128, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else if (hidden_size <= 4096) {
            rms_norm_fp8_quant_residual<scalar_in_t, scalar_out_t, T_ACC, Vec, 256, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 256, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else if (hidden_size <= 8192) {
            rms_norm_fp8_quant_residual<scalar_in_t, scalar_out_t, T_ACC, Vec, 512, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 512, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else {  // <= 16384
            rms_norm_fp8_quant_residual<scalar_in_t, scalar_out_t, T_ACC, Vec, 1024, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 1024, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          }
        } else {
          if (hidden_size <= 1024) {
            rms_norm_fp8_quant_<scalar_in_t, scalar_out_t, T_ACC, Vec, 64, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 64, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else if (hidden_size <= 2048) {
            rms_norm_fp8_quant_<scalar_in_t, scalar_out_t, T_ACC, Vec, 128, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 128, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else if (hidden_size <= 4096) {
            rms_norm_fp8_quant_<scalar_in_t, scalar_out_t, T_ACC, Vec, 256, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 256, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else if (hidden_size <= 8192) {
            rms_norm_fp8_quant_<scalar_in_t, scalar_out_t, T_ACC, Vec, 512, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 512, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          } else {  // <= 16384
            rms_norm_fp8_quant_<scalar_in_t, scalar_out_t, T_ACC, Vec, 1024, update_input, IS_GEMMA, IS_E4M3>
                <<<grid_opt, 1024, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, num_tokens);
          }
        }
      });
    });
  } else {
    dispatch_out_byte([&](auto* out_data) {
      using scalar_out_t = std::remove_pointer_t<decltype(out_data)>;

      dispatch_fp8_and_update([&](auto is_e4m3_tag, auto update_tag) {
        constexpr bool IS_E4M3 = decltype(is_e4m3_tag)::value;
        constexpr bool update_input = decltype(update_tag)::value;

        if (residual_flag) {
          if (hidden_size <= 1024) {
            rms_norm_fp8_quant_residual_loop<scalar_in_t, scalar_out_t, T_ACC, 1, 256, update_input, IS_GEMMA, IS_E4M3>
                <<<num_tokens, 256, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, res_stride, out_stride);
          } else if (hidden_size <= 4096) {
            rms_norm_fp8_quant_residual_loop<scalar_in_t, scalar_out_t, T_ACC, 1, 512, update_input, IS_GEMMA, IS_E4M3>
                <<<num_tokens, 512, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, res_stride, out_stride);
          } else {
            rms_norm_fp8_quant_residual_loop<scalar_in_t, scalar_out_t, T_ACC, 1, 1024, update_input, IS_GEMMA, IS_E4M3>
                <<<num_tokens, 1024, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, res_stride, out_stride);
          }
        } else {
          if (hidden_size <= 1024) {
            rms_norm_fp8_quant_loop<scalar_in_t, scalar_out_t, T_ACC, 1, 256, update_input, IS_GEMMA, IS_E4M3>
                <<<num_tokens, 256, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, out_stride);
          } else if (hidden_size <= 4096) {
            rms_norm_fp8_quant_loop<scalar_in_t, scalar_out_t, T_ACC, 1, 512, update_input, IS_GEMMA, IS_E4M3>
                <<<num_tokens, 512, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, out_stride);
          } else {
            rms_norm_fp8_quant_loop<scalar_in_t, scalar_out_t, T_ACC, 1, 1024, update_input, IS_GEMMA, IS_E4M3>
                <<<num_tokens, 1024, 0, stream>>>(self_data, other_data, weight_data, hidden_size, eps_f32, out_data, scaless, stride, out_stride);
          }
        }
      });
    });
  }
}

template <bool IS_GEMMA>
void rms_norm_fp8_quant_impl(
  Tensor &out,
  Tensor& input,
  const Tensor& weight,
  Tensor scales, 
  double epsilon,
  std::optional<Tensor> residual,
  bool update_input, 
  int fp8type
) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    input.scalar_type(),
    "fused_add_rms_norm_fp8_quant_kernel",
    [&] {
      fuse_rms_norm_fp8_quant<scalar_t, IS_GEMMA>(out,input,weight,scales,epsilon,residual,update_input,fp8type);
    });
}

void rms_norm_fp8_quant(
  Tensor &out, Tensor& input, const Tensor& weight, Tensor scales, 
  double epsilon, std::optional<Tensor> residual, bool update_input, int fp8type
) {
    rms_norm_fp8_quant_impl<false>(out, input, weight, scales, epsilon, residual, update_input, fp8type);
}
void gemma_rms_norm_fp8_quant(
  Tensor &out, Tensor& input, const Tensor& weight, Tensor scales, 
  double epsilon, std::optional<Tensor> residual, bool update_input, int fp8type
) {
    rms_norm_fp8_quant_impl<true>(out, input, weight, scales, epsilon, residual, update_input, fp8type);
}

}}