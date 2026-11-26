// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"

#include <stdint.h>
#include <limits>
#include <stdio.h>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <torch/extension.h>

#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAMathCompat.h>


#if defined(TORCH_VERSION_OVER_2120)
// torch>=2.12 removed C10_WARP_SIZE_STATIC, wavefront size is fixed at 64
#define C10_WARP_SIZE 64
#elif defined(TORCH_VERSION_OVER_290)
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {  
namespace native {

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))    

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

inline __device__ uint8_t float_to_fp8e4m3(float f) {
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
  
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  
  if (f_bits >= fp8_max) {
    result = 0x7f;  
  } else {
    if (f_bits < (UINT32_C(121) << 23)) {
      f_bits = c10::detail::fp32_to_bits(
        c10::detail::fp32_from_bits(f_bits) + 
        c10::detail::fp32_from_bits(denorm_mask)
      );
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint8_t mant_odd = (f_bits >> 20) & 1;
      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 20);
    }
  }
  result |= static_cast<uint8_t>(sign >> 24);
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
    int32_t out;
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
    int32_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
    return static_cast<uint32_t>(out);
  #endif
}

#if defined(__gfx946__)
// ============================================================================
// gfx946（少伯）分组量化硬件内置原语（DTK 27.04-1114 BuiltinsLingShu.def 实测勘误）
//
// 设计目标：
//   1. 适配 gfx946 每 CU 512 KB VGPR 预算：在 USE_UE8M0=true + BF16/FP16 主路径下，
//      直接以 8×uint32（raw_vec[8]）完成 16 元素打包 absmax 与一步缩放转 FP8，
//      彻底消除 16 个 FP32 VGPR（input_float_vec[16]），显著提升 CU 常驻 wave 数。
//   2. 消除 1/4-rate SPF 瓶颈：用硬件 v_cvt_pkrp_e8m0_f32（向 +inf 舍入即 ceil）与
//      v_cvt_pk_f32_e8m0 取代软件 log2f + ceilf + exp2f + 1.0f/s；用硬件
//      v_cvt_scale_pk_{fp8,bf8}_{bf16,f16} 直接执行 x / S1.E8M0（1-Pass，无 mov_preserve）。
//   3. 跨半波归约优化：stride=16/32 采用双 "+v" 内联汇编 v_permlane{16,32}_swap_b32，
//      走 1-Pass VALU 寄存器互换替代 DS 管线的 ds_bpermute_b32（消除 lgkmcnt 等待）。
// ============================================================================
using hcu_quant_v2f = float __attribute__((ext_vector_type(2)));
using hcu_quant_v2s = short __attribute__((ext_vector_type(2)));
using hcu_quant_v2h = _Float16 __attribute__((ext_vector_type(2)));

// BF16 打包绝对值极大：先按位清零高低 16-bit 符号位，再调用硬件 v_pk_max_bf16。
// 勘误：DTK 27.04 BuiltinsLingShu.def 必须传 op_sel_lo=0, op_sel_hi=7 才选择 {lo,lo}/{hi,hi}；
// 传 (0,0) 会误产出 op_sel_hi:[0,0] 导致高半字丢失。
__device__ __forceinline__ uint32_t hcu_pk_absmax_bf16_u32(uint32_t a_u32, uint32_t b_u32) {
  uint32_t a_abs = a_u32 & 0x7FFF7FFFU;
  uint32_t b_abs = b_u32 & 0x7FFF7FFFU;
  hcu_quant_v2s va = __builtin_bit_cast(hcu_quant_v2s, a_abs);
  hcu_quant_v2s vb = __builtin_bit_cast(hcu_quant_v2s, b_abs);
  hcu_quant_v2s vm = __builtin_hcu_pk_max_bf16(va, vb, 0, 7);
  return __builtin_bit_cast(uint32_t, vm);
}

// 将 2×BF16 非负打包对展开为 2 个 FP32（低/高 16 位对齐到 FP32 高 16 位）并取较大值。
__device__ __forceinline__ float hcu_bf16_pair_max_to_f32(uint32_t pair_abs_u32) {
  uint32_t lo_f32_bits = (pair_abs_u32 & 0xFFFFU) << 16;
  uint32_t hi_f32_bits = pair_abs_u32 & 0xFFFF0000U;
  return fmaxf(__uint_as_float(lo_f32_bits), __uint_as_float(hi_f32_bits));
}

// FP16 打包绝对值极大：清零符号位后直接发射 VOP3P v_pk_max_f16。
// 勘误：DTK 27.04 头文件中 __hmax2 仅对 __hip_bfloat162 重载，不支持 __half2，故用单条汇编直出。
__device__ __forceinline__ uint32_t hcu_pk_absmax_f16_u32(uint32_t a_u32, uint32_t b_u32) {
  uint32_t a_abs = a_u32 & 0x7FFF7FFFU;
  uint32_t b_abs = b_u32 & 0x7FFF7FFFU;
  uint32_t res;
  asm("v_pk_max_f16 %0, %1, %2" : "=v"(res) : "v"(a_abs), "v"(b_abs));
  return res;
}

__device__ __forceinline__ float hcu_f16_pair_max_to_f32(uint32_t pair_abs_u32) {
  __half2 v = __builtin_bit_cast(__half2, pair_abs_u32);
  float2 f = __half22float2(v);
  return fmaxf(f.x, f.y);
}

// 单指令将 scale_raw 向 +inf 舍入为 8-bit E8M0（等价于 exp2f(ceilf(log2f(scale_raw))) 的指数编码），
// 结果落在返回 int 的 byte 0（即 byte_sel=0）。
__device__ __forceinline__ int hcu_f32_to_e8m0_rp(float scale_raw) {
  return __builtin_hcu_cvt_pkrp_e8m0_f32(scale_raw, 0.0f, 0, false);
}

// 将打包 E8M0 的 byte 0 还原为精确 2 的幂次 FP32 scale（含 E8M0=0 -> 次正规 2^-127）。
__device__ __forceinline__ float hcu_e8m0_to_f32(int e8m0_packed) {
  hcu_quant_v2f pair = __builtin_hcu_cvt_pk_f32_e8m0(e8m0_packed, false, 0, false);
  return pair[0];
}

// 4×BF16 -> 4×FP8（E4M3/E5M2）硬件融合缩放转换：
// - ISA v1.8 语义：bf16_to_fp8(S0 / S1.E8M0)，方向为除法，无需计算 1/scale；
// - 第 1 次写入 dst[15:0]（dst_sel=false），第 2 次写入 dst[31:16]（dst_sel=true 保留低 16 位）。
template <bool IS_E4M3>
__device__ __forceinline__ uint32_t hcu_bf16x4_scale_to_fp8x4(
    uint32_t bf16_01, uint32_t bf16_23, int e8m0_packed) {
  hcu_quant_v2s p01 = __builtin_bit_cast(hcu_quant_v2s, bf16_01);
  hcu_quant_v2s p23 = __builtin_bit_cast(hcu_quant_v2s, bf16_23);
  int32_t res = 0;
  if constexpr (IS_E4M3) {
    res = __builtin_hcu_cvt_scale_pk_fp8_bf16(p01, e8m0_packed, res, false, 0);
    res = __builtin_hcu_cvt_scale_pk_fp8_bf16(p23, e8m0_packed, res, true, 0);
  } else {
    res = __builtin_hcu_cvt_scale_pk_bf8_bf16(p01, e8m0_packed, res, false, 0);
    res = __builtin_hcu_cvt_scale_pk_bf8_bf16(p23, e8m0_packed, res, true, 0);
  }
  return static_cast<uint32_t>(res);
}

// 4×FP16 -> 4×FP8（E4M3/E5M2）硬件融合缩放转换，语义同上。
template <bool IS_E4M3>
__device__ __forceinline__ uint32_t hcu_f16x4_scale_to_fp8x4(
    uint32_t f16_01, uint32_t f16_23, int e8m0_packed) {
  hcu_quant_v2h p01 = __builtin_bit_cast(hcu_quant_v2h, f16_01);
  hcu_quant_v2h p23 = __builtin_bit_cast(hcu_quant_v2h, f16_23);
  int32_t res = 0;
  if constexpr (IS_E4M3) {
    res = __builtin_hcu_cvt_scale_pk_fp8_f16(p01, e8m0_packed, res, false, 0);
    res = __builtin_hcu_cvt_scale_pk_fp8_f16(p23, e8m0_packed, res, true, 0);
  } else {
    res = __builtin_hcu_cvt_scale_pk_bf8_f16(p01, e8m0_packed, res, false, 0);
    res = __builtin_hcu_cvt_scale_pk_bf8_f16(p23, e8m0_packed, res, true, 0);
  }
  return static_cast<uint32_t>(res);
}

// 4×FP32 -> 4×FP8 硬件 E8M0 缩放转换。
// 跨 DTK 27.04 小版本兼容 + 勘误：
//   - DTK 27.04-1114 将原 6 参 builtin 拆分为 5 参（无 init_src2）+ 6 参 *_wsrc2，
//     且 *_wsrc2 存在编译器 lowering bug（丢弃 init_src2）；
//   - DTK 27.04-1010 仅有 6 参版且在 dst_sel=true 串联时同样会丢弃前次写入结果。
// 因此两版本统一对每对 (a,b)/(c,d) 以 dst_sel=false 独立转换并用位运算拼装 4×FP8，
// 并以 __has_builtin(__builtin_hcu_cvt_scale_pk_fp8_f32_wsrc2) 区分 5 参/6 参原型。
template <bool IS_E4M3>
__device__ __forceinline__ uint32_t hcu_f32x4_scale_to_fp8x4(
    float a, float b, float c, float d, int e8m0_packed) {
  uint32_t lo, hi;
#if defined(__has_builtin) && __has_builtin(__builtin_hcu_cvt_scale_pk_fp8_f32_wsrc2)
  if constexpr (IS_E4M3) {
    lo = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_fp8_f32(a, b, e8m0_packed, false, 0));
    hi = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_fp8_f32(c, d, e8m0_packed, false, 0));
  } else {
    lo = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_bf8_f32(a, b, e8m0_packed, false, 0));
    hi = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_bf8_f32(c, d, e8m0_packed, false, 0));
  }
#else
  if constexpr (IS_E4M3) {
    lo = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_fp8_f32(a, b, e8m0_packed, 0, false, 0));
    hi = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_fp8_f32(c, d, e8m0_packed, 0, false, 0));
  } else {
    lo = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_bf8_f32(a, b, e8m0_packed, 0, false, 0));
    hi = static_cast<uint32_t>(__builtin_hcu_cvt_scale_pk_bf8_f32(c, d, e8m0_packed, 0, false, 0));
  }
#endif
  return (lo & 0xFFFFU) | (hi << 16);
}

// 非 UE8M0 任意 FP32 scale 路径：利用 ext_vector_type(2) 在 -O3 下自动生成 VOP3P v_pk_mul_f32，
// 将 16 次标量乘 inv_s 折半为 8 条打包乘，再走硬件 v_cvt_pk_{fp8,bf8}_f32。
template <bool IS_E4M3>
__device__ __forceinline__ uint32_t hcu_f32x4_mul_to_fp8x4(
    float a, float b, float c, float d, float inv_s) {
  hcu_quant_v2f v01 = {a, b};
  hcu_quant_v2f v23 = {c, d};
  hcu_quant_v2f vs = {inv_s, inv_s};
  v01 = v01 * vs;
  v23 = v23 * vs;
  if constexpr (IS_E4M3) {
    return float_to_fp8e4m3_x4(v01[0], v01[1], v23[0], v23[1]);
  } else {
    return float_to_fp8e5m2_x4(v01[0], v01[1], v23[0], v23[1]);
  }
}

// 跨 16/32 lane 的 1-Pass VALU 极大值归约：
// 勘误：DTK 27.04 内置 __builtin_hcu_permlane{16,32}_swap_b32 前端签名缺失第 2 路写回，
// 故采用双 "+v" 读写约束内联汇编：初始令 other = val，交换后偶/奇行分别持有自身与对端值，
// fmaxf(val, other) 在全部 active lane 上严格等价于 fmaxf(val, shfl_xor(val, 16/32))。
__device__ __forceinline__ float hcu_permlane16_swap_fmax(float val) {
  float other = val;
  asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return fmaxf(val, other);
}

__device__ __forceinline__ float hcu_permlane32_swap_fmax(float val) {
  float other = val;
  asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(val), "+v"(other));
  return fmaxf(val, other);
}
#endif

template <typename T,int reducesize=C10_WARP_SIZE>
__inline__ __device__ T WarpReduceMax_ROW(T val) {
#if defined(__gfx946__)
  if constexpr (std::is_same_v<T, float> && reducesize == 64) {
    val = hcu_permlane32_swap_fmax(val);
    val = hcu_permlane16_swap_fmax(val);
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
  }
#endif
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
    val =  fmaxf(val,  __shfl_down(val, offset));
  }
  return val;
}

template <typename T,int block_size=512>
__inline__ __device__ T BlockReduceMax_ROW(T val, T* shared) {
  constexpr int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceMax_ROW<T>(val);
  if constexpr(block_size==C10_WARP_SIZE)
  {
    return val;
  }
  else{
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0&&wid<share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0&&lid<share_size) {
      val = WarpReduceMax_ROW<T,share_size>(shared[lid]);
    }
    return val;
  }
}

template <int THREADS_PER_GROUP>
__inline__ __device__ float WarpGroupReduceMax(float val) {
#if defined(__gfx946__)
    if constexpr (THREADS_PER_GROUP == 32) {
        val = hcu_permlane16_swap_fmax(val);
        #pragma unroll
        for (int mask = 8; mask > 0; mask >>= 1) {
            val = fmaxf(val, __shfl_xor(val, mask));
        }
        return val;
    }
#endif
    #pragma unroll
    for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor(val, mask));
    }
    return val;
}

// ============================================================================
// 小分组量化内核（group_size < 1024，每线程负责 VEC=16 个连续元素）
// - THREADS_PER_GROUP = group_size / 16 ∈ {1, 2, 4, 8, 16, 32}；典型 128 对应 8 线程/组
// - USE_UE8M0 编译期特化：
//   * gfx946 + USE_UE8M0=true + BF16/FP16：8×u32 寄存器直算打包 absmax + 硬件 E8M0 +
//     硬件 cvt_scale_pk 一步量化，零 input_float_vec[16]（省 16 VGPR，消除 4 条 SPF 指令）
//   * gfx946 + USE_UE8M0=true + FP32：硬件 E8M0 + 非 _wsrc2 缩放直转 FP8
//   * gfx946 + USE_UE8M0=false：v2f 自动向量化 v_pk_mul_f32 + 硬件 cvt_pk_fp8_f32
//   * 非 gfx946：保持原路径，修复 int32_t res 未初始化与 int64_t group 索引溢出
// ============================================================================
template <typename scalar_t, int THREADS_PER_GROUP, bool is_e4m3, bool USE_UE8M0>
__global__ void per_token_group_quant_fp8_small_kernel(
    const scalar_t* __restrict__ input, 
    uint8_t* __restrict__ output,
    float* scale_out, 
    const int group_size,
    const float eps,
    const int64_t total_groups) {

    constexpr int VEC = 16;

    int64_t global_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t global_group_id = global_tid / THREADS_PER_GROUP;
    int lane_in_group = static_cast<int>(global_tid % THREADS_PER_GROUP);

    if (global_group_id >= total_groups) return;

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecFp8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    int64_t idx = global_group_id * group_size + lane_in_group * VEC;
    constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;
    constexpr float INV_FP8_MAX_VAL = is_e4m3 ? (1.0f / 448.0f) : (1.0f / 57344.0f);

#if defined(__gfx946__)
    if constexpr (USE_UE8M0 && (std::is_same_v<scalar_t, at::BFloat16> || std::is_same_v<scalar_t, at::Half>)) {
        uint32_t raw_vec[8];
        *(VecType*)raw_vec = *(const VecType*)(input + idx);

        uint32_t m0, m1, m2, m3, m01, m23, m_pair;
        float max_val;
        if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            m0 = hcu_pk_absmax_bf16_u32(raw_vec[0], raw_vec[1]);
            m1 = hcu_pk_absmax_bf16_u32(raw_vec[2], raw_vec[3]);
            m2 = hcu_pk_absmax_bf16_u32(raw_vec[4], raw_vec[5]);
            m3 = hcu_pk_absmax_bf16_u32(raw_vec[6], raw_vec[7]);
            m01 = hcu_pk_absmax_bf16_u32(m0, m1);
            m23 = hcu_pk_absmax_bf16_u32(m2, m3);
            m_pair = hcu_pk_absmax_bf16_u32(m01, m23);
            max_val = hcu_bf16_pair_max_to_f32(m_pair);
        } else {
            m0 = hcu_pk_absmax_f16_u32(raw_vec[0], raw_vec[1]);
            m1 = hcu_pk_absmax_f16_u32(raw_vec[2], raw_vec[3]);
            m2 = hcu_pk_absmax_f16_u32(raw_vec[4], raw_vec[5]);
            m3 = hcu_pk_absmax_f16_u32(raw_vec[6], raw_vec[7]);
            m01 = hcu_pk_absmax_f16_u32(m0, m1);
            m23 = hcu_pk_absmax_f16_u32(m2, m3);
            m_pair = hcu_pk_absmax_f16_u32(m01, m23);
            max_val = hcu_f16_pair_max_to_f32(m_pair);
        }

        max_val = WarpGroupReduceMax<THREADS_PER_GROUP>(max_val);
        float scale_raw = fmaxf(max_val, eps) * INV_FP8_MAX_VAL;
        int e8m0_packed = hcu_f32_to_e8m0_rp(scale_raw);

        if (lane_in_group == 0) {
            scale_out[global_group_id] = (scale_raw == 0.f) ? 0.f : hcu_e8m0_to_f32(e8m0_packed);
        }

        uint32_t out_u32[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                out_u32[p] = hcu_bf16x4_scale_to_fp8x4<is_e4m3>(
                    raw_vec[2 * p], raw_vec[2 * p + 1], e8m0_packed);
            } else {
                out_u32[p] = hcu_f16x4_scale_to_fp8x4<is_e4m3>(
                    raw_vec[2 * p], raw_vec[2 * p + 1], e8m0_packed);
            }
        }
        *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_u32;
        return;
    }
#endif

    scalar_t input_vec[VEC];
    *(VecType*)input_vec = *(const VecType*)(input + idx);

    float input_float_vec[VEC];
    float max_val = 0.f;

    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
        input_float_vec[ii] = static_cast<float>(input_vec[ii]);
        max_val = fmaxf(max_val, fabsf(input_float_vec[ii]));
    }

    max_val = WarpGroupReduceMax<THREADS_PER_GROUP>(max_val);
    max_val = fmaxf(max_val, eps);

#if defined(__gfx946__)
    if constexpr (USE_UE8M0) {
        float scale_raw = max_val * INV_FP8_MAX_VAL;
        int e8m0_packed = hcu_f32_to_e8m0_rp(scale_raw);
        if (lane_in_group == 0) {
            scale_out[global_group_id] = (scale_raw == 0.f) ? 0.f : hcu_e8m0_to_f32(e8m0_packed);
        }
        uint32_t out_u32[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            out_u32[p] = hcu_f32x4_scale_to_fp8x4<is_e4m3>(
                input_float_vec[4 * p + 0], input_float_vec[4 * p + 1],
                input_float_vec[4 * p + 2], input_float_vec[4 * p + 3],
                e8m0_packed);
        }
        *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_u32;
        return;
    } else {
        float s_group_scale = max_val * INV_FP8_MAX_VAL;
        if (lane_in_group == 0) {
            scale_out[global_group_id] = s_group_scale;
        }
        float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;
        uint32_t out_u32[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            out_u32[p] = hcu_f32x4_mul_to_fp8x4<is_e4m3>(
                input_float_vec[4 * p + 0], input_float_vec[4 * p + 1],
                input_float_vec[4 * p + 2], input_float_vec[4 * p + 3],
                inv_s);
        }
        *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_u32;
        return;
    }
#else
    float scale_raw = max_val / FP8_MAX_VAL;
    float s_group_scale;
    if constexpr (USE_UE8M0) {
        s_group_scale = exp2f(ceilf(log2f(scale_raw)));
    } else {
        s_group_scale = scale_raw;
    }

    if (lane_in_group == 0) {
        scale_out[global_group_id] = s_group_scale;
    }

    float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;

    uint8_t out_vec[VEC];
    #pragma unroll
    for (int ii = 0; ii < VEC; ii += 4) {
        uint32_t packed;
        if constexpr (is_e4m3) {
            packed = float_to_fp8e4m3_x4(
                input_float_vec[ii] * inv_s,
                input_float_vec[ii+1] * inv_s,
                input_float_vec[ii+2] * inv_s,
                input_float_vec[ii+3] * inv_s);
        } else {
            packed = float_to_fp8e5m2_x4(
                input_float_vec[ii] * inv_s,
                input_float_vec[ii+1] * inv_s,
                input_float_vec[ii+2] * inv_s,
                input_float_vec[ii+3] * inv_s);
        }
        *(uint32_t*)(&out_vec[ii]) = packed;
    }

    *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_vec;
#endif
}

// ============================================================================
// 大分组量化内核（group_size >= 1024，每 Block 处理 1 个 group，block_size = group_size / 16）
// - BlockReduceMax_ROW 在 gfx946 上通过 v_permlane{32,16}_swap_b32 替换 stride 32/16 的
//   ds_bpermute，消除跨半波 LDS 往返
// - tid == 0 计算 E8M0 编码并经 __shared__ s_e8m0_packed 广播给同块所有线程
// ============================================================================
template <typename scalar_t, int block_size, bool is_e4m3, bool USE_UE8M0>
__global__ void per_token_group_quant_fp8_kernel(
    const scalar_t* __restrict__ input, 
    uint8_t* __restrict__ output,
    float* scale_out, 
    const int group_size,   
    const float eps) { 

    constexpr int VEC = 16;
    int64_t bid = static_cast<int64_t>(blockIdx.x);

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    __shared__ float s_group_scale;
#if defined(__gfx946__)
    __shared__ int s_e8m0_packed;
#endif

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecFp8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    int tid = threadIdx.x;
    int tcol = group_size / VEC;

    int64_t idx = (bid * tcol + tid) * VEC;
    constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;
    constexpr float INV_FP8_MAX_VAL = is_e4m3 ? (1.0f / 448.0f) : (1.0f / 57344.0f);

#if defined(__gfx946__)
    if constexpr (USE_UE8M0 && (std::is_same_v<scalar_t, at::BFloat16> || std::is_same_v<scalar_t, at::Half>)) {
        uint32_t raw_vec[8];
        float max_val = 0.f;
        if (tid < tcol) {
            *(VecType*)raw_vec = *(const VecType*)(input + idx);
            uint32_t m0, m1, m2, m3, m01, m23, m_pair;
            if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                m0 = hcu_pk_absmax_bf16_u32(raw_vec[0], raw_vec[1]);
                m1 = hcu_pk_absmax_bf16_u32(raw_vec[2], raw_vec[3]);
                m2 = hcu_pk_absmax_bf16_u32(raw_vec[4], raw_vec[5]);
                m3 = hcu_pk_absmax_bf16_u32(raw_vec[6], raw_vec[7]);
                m01 = hcu_pk_absmax_bf16_u32(m0, m1);
                m23 = hcu_pk_absmax_bf16_u32(m2, m3);
                m_pair = hcu_pk_absmax_bf16_u32(m01, m23);
                max_val = hcu_bf16_pair_max_to_f32(m_pair);
            } else {
                m0 = hcu_pk_absmax_f16_u32(raw_vec[0], raw_vec[1]);
                m1 = hcu_pk_absmax_f16_u32(raw_vec[2], raw_vec[3]);
                m2 = hcu_pk_absmax_f16_u32(raw_vec[4], raw_vec[5]);
                m3 = hcu_pk_absmax_f16_u32(raw_vec[6], raw_vec[7]);
                m01 = hcu_pk_absmax_f16_u32(m0, m1);
                m23 = hcu_pk_absmax_f16_u32(m2, m3);
                m_pair = hcu_pk_absmax_f16_u32(m01, m23);
                max_val = hcu_f16_pair_max_to_f32(m_pair);
            }
        }
        max_val = BlockReduceMax_ROW<float, block_size>(max_val, val_shared);
        if (tid == 0) {
            float scale_raw = fmaxf(max_val, eps) * INV_FP8_MAX_VAL;
            int e8m0_packed = hcu_f32_to_e8m0_rp(scale_raw);
            s_e8m0_packed = e8m0_packed;
            scale_out[bid] = (scale_raw == 0.f) ? 0.f : hcu_e8m0_to_f32(e8m0_packed);
        }
        __syncthreads();
        if (tid < tcol) {
            int e8m0_packed = s_e8m0_packed;
            uint32_t out_u32[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                    out_u32[p] = hcu_bf16x4_scale_to_fp8x4<is_e4m3>(
                        raw_vec[2 * p], raw_vec[2 * p + 1], e8m0_packed);
                } else {
                    out_u32[p] = hcu_f16x4_scale_to_fp8x4<is_e4m3>(
                        raw_vec[2 * p], raw_vec[2 * p + 1], e8m0_packed);
                }
            }
            *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_u32;
        }
        return;
    }
#endif

    float input_float_vec[VEC];
    float max_val = 0.f;
    if (tid < tcol) {
        scalar_t input_vec[VEC];
        *(VecType*)input_vec = *(const VecType*)(input + idx);
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            input_float_vec[ii] = static_cast<float>(input_vec[ii]);
            max_val = fmaxf(max_val, fabsf(input_float_vec[ii]));
        }
    }

    max_val = BlockReduceMax_ROW<float, block_size>(max_val, val_shared);

#if defined(__gfx946__)
    if constexpr (USE_UE8M0) {
        if (tid == 0) {
            float scale_raw = fmaxf(max_val, eps) * INV_FP8_MAX_VAL;
            int e8m0_packed = hcu_f32_to_e8m0_rp(scale_raw);
            s_e8m0_packed = e8m0_packed;
            scale_out[bid] = (scale_raw == 0.f) ? 0.f : hcu_e8m0_to_f32(e8m0_packed);
        }
        __syncthreads();
        if (tid < tcol) {
            int e8m0_packed = s_e8m0_packed;
            uint32_t out_u32[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                out_u32[p] = hcu_f32x4_scale_to_fp8x4<is_e4m3>(
                    input_float_vec[4 * p + 0], input_float_vec[4 * p + 1],
                    input_float_vec[4 * p + 2], input_float_vec[4 * p + 3],
                    e8m0_packed);
            }
            *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_u32;
        }
        return;
    } else {
        if (tid == 0) {
            float gs = fmaxf(max_val, eps) * INV_FP8_MAX_VAL;
            s_group_scale = gs;
            scale_out[bid] = gs;
        }
        __syncthreads();
        float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;
        if (tid < tcol) {
            uint32_t out_u32[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                out_u32[p] = hcu_f32x4_mul_to_fp8x4<is_e4m3>(
                    input_float_vec[4 * p + 0], input_float_vec[4 * p + 1],
                    input_float_vec[4 * p + 2], input_float_vec[4 * p + 3],
                    inv_s);
            }
            *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_u32;
        }
        return;
    }
#else
    if (tid == 0) {
        max_val = fmaxf(max_val, eps);
        float scale_raw = max_val / FP8_MAX_VAL;
        if constexpr (USE_UE8M0) {
            s_group_scale = exp2f(ceilf(log2f(scale_raw)));
        } else {
            s_group_scale = scale_raw;
        }
        scale_out[bid] = s_group_scale;
    }
    __syncthreads();

    float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;

    uint8_t out_vec[VEC];
    if (tid < tcol) {
        #pragma unroll
        for (int ii = 0; ii < VEC; ii += 4) {
            uint32_t packed;
            if constexpr (is_e4m3) {
                packed = float_to_fp8e4m3_x4(
                    input_float_vec[ii] * inv_s,
                    input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s,
                    input_float_vec[ii+3] * inv_s);
            } else {
                packed = float_to_fp8e5m2_x4(
                    input_float_vec[ii] * inv_s,
                    input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s,
                    input_float_vec[ii+3] * inv_s);
            }
            *(uint32_t*)(&out_vec[ii]) = packed;
        }
        *(VecFp8Type*)(output + idx) = *(VecFp8Type*)out_vec;
    }
#endif
}

void per_token_group_quant_fp8(
    torch::Tensor& out,        
    torch::Tensor  input, 
    torch::Tensor& scales,
    int group_size,            
    float eps = 1e-5f,         
    bool use_ue8m0 = false)    
{
    input = input.is_contiguous() ? input : input.contiguous();
    TORCH_CHECK(out.scalar_type() == torch::kFloat8_e4m3fn || out.scalar_type() == torch::kFloat8_e5m2, 
                "Output tensor must be of type float8_e4m3fn or float8_e5m2");
 
    bool is_e4m3 = (out.scalar_type() == torch::kFloat8_e4m3fn);
    int const hidden_size = input.size(-1);
    
    TORCH_CHECK(hidden_size % group_size == 0, "hidden_size must be a multiple of group_size");
    TORCH_CHECK(group_size <= 16384, "group_size exceeds maximum supported size of 16384");
    TORCH_CHECK(group_size >= 16 && group_size % 16 == 0, "group_size must be >=16 and a multiple of 16 for vectorized VEC=16 memory access");
    TORCH_CHECK((group_size & (group_size - 1)) == 0, "group_size must be a power of 2 for optimized reductions");

    int64_t num_groups = input.numel() / group_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto out_ptr = reinterpret_cast<uint8_t*>(out.data_ptr());

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_fp8_kernel", [&] {
            auto dispatch = [&](auto is_e4m3_tag, auto ue8m0_tag) {
                constexpr bool kIsE4M3 = decltype(is_e4m3_tag)::value;
                constexpr bool kUseUe8m0 = decltype(ue8m0_tag)::value;

                if (group_size < 1024) {
                    int threads_per_group = group_size / 16;
                    constexpr int BLOCK_SIZE = 256;
                    int groups_per_block = BLOCK_SIZE / threads_per_group;
                    int num_blocks = static_cast<int>((num_groups + groups_per_block - 1) / groups_per_block);

                    auto launch_small = [&](auto tpg_tag) {
                        constexpr int kThreads = decltype(tpg_tag)::value;
                        per_token_group_quant_fp8_small_kernel<scalar_t, kThreads, kIsE4M3, kUseUe8m0>
                            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                                input.data_ptr<scalar_t>(), out_ptr, scales.data_ptr<float>(),
                                group_size, eps, num_groups);
                    };

                    if (threads_per_group == 1)       { launch_small(std::integral_constant<int, 1>{}); }
                    else if (threads_per_group == 2)  { launch_small(std::integral_constant<int, 2>{}); }
                    else if (threads_per_group == 4)  { launch_small(std::integral_constant<int, 4>{}); }
                    else if (threads_per_group == 8)  { launch_small(std::integral_constant<int, 8>{}); }
                    else if (threads_per_group == 16) { launch_small(std::integral_constant<int, 16>{}); }
                    else if (threads_per_group == 32) { launch_small(std::integral_constant<int, 32>{}); }
                } else {
                    int block_size = group_size / 16;

                    auto launch_large = [&](auto bs_tag) {
                        constexpr int kBlockSize = decltype(bs_tag)::value;
                        per_token_group_quant_fp8_kernel<scalar_t, kBlockSize, kIsE4M3, kUseUe8m0>
                            <<<num_groups, kBlockSize, 0, stream>>>(
                                input.data_ptr<scalar_t>(), out_ptr, scales.data_ptr<float>(),
                                group_size, eps);
                    };

                    if (block_size == 64)        { launch_large(std::integral_constant<int, 64>{}); }
                    else if (block_size == 128)  { launch_large(std::integral_constant<int, 128>{}); }
                    else if (block_size == 256)  { launch_large(std::integral_constant<int, 256>{}); }
                    else if (block_size == 512)  { launch_large(std::integral_constant<int, 512>{}); }
                    else if (block_size == 1024) { launch_large(std::integral_constant<int, 1024>{}); }
                }
            };

            if (is_e4m3) {
                if (use_ue8m0) { dispatch(std::true_type{}, std::true_type{}); }
                else           { dispatch(std::true_type{}, std::false_type{}); }
            } else {
                if (use_ue8m0) { dispatch(std::false_type{}, std::true_type{}); }
                else           { dispatch(std::false_type{}, std::false_type{}); }
            }
        }
    );
}

template <typename scalar_t, int THREADS_PER_GROUP, bool is_e4m3>
__global__ void fused_store_index_k_cache_small_kernel(
    const scalar_t* __restrict__ input,       
    uint8_t* __restrict__ buf_ptr,           
    const int64_t* __restrict__ loc_ptr,      
    const int hidden_size,
    const int page_size,
    const int64_t buf_numel_per_page,      
    const int64_t s_offset_nbytes_in_page,    
    const float eps,
    const bool use_ue8m0,
    const int64_t num_tokens) {

    constexpr int VEC = 16;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int global_token_id = global_tid / THREADS_PER_GROUP;
    int lane_in_group = global_tid % THREADS_PER_GROUP;

    if (global_token_id >= num_tokens) return;


    int64_t loc = loc_ptr[global_token_id];
    int64_t page_idx = loc / page_size;
    int64_t token_offset = loc % page_size;

    uint8_t* out_k_ptr = buf_ptr + page_idx * buf_numel_per_page + token_offset * hidden_size;
    
    float* out_s_ptr = reinterpret_cast<float*>(buf_ptr + page_idx * buf_numel_per_page + s_offset_nbytes_in_page) + token_offset;

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecFp8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    int64_t idx = (int64_t)global_token_id * hidden_size + lane_in_group * VEC;

    scalar_t input_vec[VEC];
    *(VecType*)input_vec = *(VecType*)(input + idx);

    float input_float_vec[VEC];
    float max_val = 0.f;
    
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
        input_float_vec[ii] = static_cast<float>(input_vec[ii]);
        max_val = fmaxf(max_val, fabsf(input_float_vec[ii]));
    }

    max_val = WarpGroupReduceMax<THREADS_PER_GROUP>(max_val);

    constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;
    max_val = fmaxf(max_val, eps);
    float scale_raw = max_val / FP8_MAX_VAL;
    
    float s_group_scale;
    if (use_ue8m0) {
        s_group_scale = exp2f(ceilf(log2f(scale_raw)));
    } else {
        s_group_scale = scale_raw;
    }
    if (lane_in_group == 0) {
        *out_s_ptr = s_group_scale;
    }

    float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;
    uint8_t out_vec[VEC];
    
    #pragma unroll
    for(int ii = 0; ii < VEC; ii+=4){
        uint32_t packed;
        if constexpr (is_e4m3) {
            packed = float_to_fp8e4m3_x4(
                input_float_vec[ii] * inv_s, input_float_vec[ii+1] * inv_s,
                input_float_vec[ii+2] * inv_s, input_float_vec[ii+3] * inv_s);
        } else {
            packed = float_to_fp8e5m2_x4(
                input_float_vec[ii] * inv_s, input_float_vec[ii+1] * inv_s,
                input_float_vec[ii+2] * inv_s, input_float_vec[ii+3] * inv_s);
        }
        *(uint32_t*)(&out_vec[ii]) = packed;
    }
    *(VecFp8Type*)(out_k_ptr + lane_in_group * VEC) = *(VecFp8Type*)out_vec;
}

void fuse_act_quant_and_store_index_k_cache(
    torch::Tensor input,     
    torch::Tensor buf,        
    torch::Tensor loc,         
    int page_size,
    float eps = 1e-5f,         
    bool use_ue8m0 = false,
    bool is_e4m3 = true)       
{
    input = input.is_contiguous() ? input : input.contiguous();
    loc = loc.is_contiguous() ? loc : loc.contiguous();
    
    
    TORCH_CHECK(loc.scalar_type() == torch::kInt64, "loc tensor must be int64 to align with Triton");

    int const hidden_size = input.size(-1);
    int64_t num_tokens = input.numel() / hidden_size;
    
    TORCH_CHECK(hidden_size <= 16384, "hidden_size exceeds 16384");
    TORCH_CHECK(hidden_size >= 16 && hidden_size % 16 == 0, "hidden_size must be multiple of 16");

    int64_t s_offset_nbytes_in_page = page_size * hidden_size; 
    int64_t buf_numel_per_page = page_size * hidden_size + page_size * sizeof(float);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto buf_ptr = reinterpret_cast<uint8_t*>(buf.data_ptr());
    
    auto loc_ptr = loc.data_ptr<int64_t>();

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "fused_store_index_k_cache_small_kernel", [&] {
            int threads_per_group = hidden_size / 16;
            constexpr int BLOCK_SIZE = 256; 
            int groups_per_block = BLOCK_SIZE / threads_per_group;
            int num_blocks = (num_tokens + groups_per_block - 1) / groups_per_block;

            #define LAUNCH_SMALL_FUSED_KERNEL(THREADS, IS_E4M3_BOOL)                       \
                fused_store_index_k_cache_small_kernel<scalar_t, THREADS, IS_E4M3_BOOL>    \
                    <<<num_blocks, BLOCK_SIZE, 0, stream>>>(                               \
                        input.data_ptr<scalar_t>(), buf_ptr, loc_ptr,                      \
                        hidden_size, page_size, buf_numel_per_page,                        \
                        s_offset_nbytes_in_page, eps, use_ue8m0, num_tokens)

            if (is_e4m3) {
                if (threads_per_group == 1) { LAUNCH_SMALL_FUSED_KERNEL(1, true); }
                else if (threads_per_group == 2) { LAUNCH_SMALL_FUSED_KERNEL(2, true); }
                else if (threads_per_group == 4) { LAUNCH_SMALL_FUSED_KERNEL(4, true); }
                else if (threads_per_group == 8) { LAUNCH_SMALL_FUSED_KERNEL(8, true); }
                else if (threads_per_group == 16) { LAUNCH_SMALL_FUSED_KERNEL(16, true); }
                else if (threads_per_group == 32) { LAUNCH_SMALL_FUSED_KERNEL(32, true); }
            } else {
                if (threads_per_group == 1) { LAUNCH_SMALL_FUSED_KERNEL(1, false); }
                else if (threads_per_group == 2) { LAUNCH_SMALL_FUSED_KERNEL(2, false); }
                else if (threads_per_group == 4) { LAUNCH_SMALL_FUSED_KERNEL(4, false); }
                else if (threads_per_group == 8) { LAUNCH_SMALL_FUSED_KERNEL(8, false); }
                else if (threads_per_group == 16) { LAUNCH_SMALL_FUSED_KERNEL(16, false); }
                else if (threads_per_group == 32) { LAUNCH_SMALL_FUSED_KERNEL(32, false); }
            }
        }
    );
}

__inline__ __device__ float DynamicWarpGroupReduceMax(float val, int threads_per_group) {
    #pragma unroll 1
    for (int mask = threads_per_group / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor(val, mask));
    }
    return val;
}

template <typename scalar_t, bool is_e4m3>
__global__ void fused_qk_quant_weight_mul_and_store_index_k_cache(
    const scalar_t* __restrict__ q_input,     // [num_q_elements, q_hidden_size]
    uint8_t* __restrict__ q_out,              // [num_q_elements, q_hidden_size]
    float* __restrict__ q_scale_out,          // [num_q_elements]
    const scalar_t* __restrict__ weights_in,  // [num_q_elements] 
    scalar_t* __restrict__ weights_out,       // [num_q_elements]
    const float q_scale_factor,             
    const float k_scale_factor,              
    const int q_hidden_size,
    const int q_threads_per_group,
    const scalar_t* __restrict__ k_input,     
    uint8_t* __restrict__ k_buf_ptr,          
    const int64_t* __restrict__ k_loc_ptr,  
    const int k_hidden_size,
    const int k_threads_per_group,
    const int page_size,
    const int64_t buf_numel_per_page,
    const int64_t s_offset_nbytes_in_page,
    const float eps,
    const bool use_ue8m0,
    const int64_t num_q_elements, 
    const int64_t num_tokens,     
    const int q_blocks                     
) {
    constexpr int VEC = 16;
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecFp8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    if (blockIdx.x < q_blocks) {
        int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        int global_q_id = global_tid / q_threads_per_group;
        int lane_in_group = global_tid % q_threads_per_group;

        if (global_q_id >= num_q_elements) return;

        int64_t idx = (int64_t)global_q_id * q_hidden_size + lane_in_group * VEC;

        scalar_t input_vec[VEC];
        *(VecType*)input_vec = *(VecType*)(q_input + idx);

        float input_float_vec[VEC];
        float max_val = 0.f;
        
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            input_float_vec[ii] = static_cast<float>(input_vec[ii]) * q_scale_factor;
            max_val = fmaxf(max_val, fabsf(input_float_vec[ii]));
        }

        max_val = DynamicWarpGroupReduceMax(max_val, q_threads_per_group);

        constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;
        max_val = fmaxf(max_val, eps);
        float scale_raw = max_val / FP8_MAX_VAL;
        
        float s_group_scale = use_ue8m0 ? exp2f(ceilf(log2f(scale_raw))) : scale_raw;

        if (lane_in_group == 0) {
            q_scale_out[global_q_id] = s_group_scale;
            if (weights_in != nullptr && weights_out != nullptr) {
                float w = static_cast<float>(weights_in[global_q_id]);
                weights_out[global_q_id] = static_cast<scalar_t>(w * s_group_scale);
            }
        }

        float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;
        uint8_t out_vec[VEC];
        
        #pragma unroll
        for(int ii = 0; ii < VEC; ii+=4){
            uint32_t packed;
            if constexpr (is_e4m3) {
                packed = float_to_fp8e4m3_x4(
                    input_float_vec[ii] * inv_s, input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s, input_float_vec[ii+3] * inv_s);
            } else {
                packed = float_to_fp8e5m2_x4(
                    input_float_vec[ii] * inv_s, input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s, input_float_vec[ii+3] * inv_s);
            }
            *(uint32_t*)(&out_vec[ii]) = packed;
        }
        *(VecFp8Type*)(q_out + idx) = *(VecFp8Type*)out_vec;
    } else {
        int k_block_idx = blockIdx.x - q_blocks; 
        int global_tid = k_block_idx * blockDim.x + threadIdx.x;
        int global_token_id = global_tid / k_threads_per_group;
        int lane_in_group = global_tid % k_threads_per_group;

        if (global_token_id >= num_tokens) return;
    
        int64_t loc = k_loc_ptr[global_token_id];
        int64_t page_idx = loc / page_size;
        int64_t token_offset = loc % page_size;

        uint8_t* out_k_ptr = k_buf_ptr + page_idx * buf_numel_per_page + token_offset * k_hidden_size;
        float* out_s_ptr = reinterpret_cast<float*>(k_buf_ptr + page_idx * buf_numel_per_page + s_offset_nbytes_in_page) + token_offset;

        int64_t idx = (int64_t)global_token_id * k_hidden_size + lane_in_group * VEC;

        scalar_t input_vec[VEC];
        *(VecType*)input_vec = *(VecType*)(k_input + idx);

        float input_float_vec[VEC];
        float max_val = 0.f;
        
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            input_float_vec[ii] = static_cast<float>(input_vec[ii]) * k_scale_factor;
            max_val = fmaxf(max_val, fabsf(input_float_vec[ii]));
        }

        max_val = DynamicWarpGroupReduceMax(max_val, k_threads_per_group);

        constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;
        max_val = fmaxf(max_val, eps);
        float scale_raw = max_val / FP8_MAX_VAL;
        
        float s_group_scale = use_ue8m0 ? exp2f(ceilf(log2f(scale_raw))) : scale_raw;

        if (lane_in_group == 0) {
            *out_s_ptr = s_group_scale;
        }

        float inv_s = (s_group_scale == 0.f) ? 0.f : 1.0f / s_group_scale;
        uint8_t out_vec[VEC];
        
        #pragma unroll
        for(int ii = 0; ii < VEC; ii+=4){
            uint32_t packed;
            if constexpr (is_e4m3) {
                packed = float_to_fp8e4m3_x4(
                    input_float_vec[ii] * inv_s, input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s, input_float_vec[ii+3] * inv_s);
            } else {
                packed = float_to_fp8e5m2_x4(
                    input_float_vec[ii] * inv_s, input_float_vec[ii+1] * inv_s,
                    input_float_vec[ii+2] * inv_s, input_float_vec[ii+3] * inv_s);
            }
            *(uint32_t*)(&out_vec[ii]) = packed;
        }
        *(VecFp8Type*)(out_k_ptr + lane_in_group * VEC) = *(VecFp8Type*)out_vec;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fuse_qk_quant_and_store_index_k_cache(
    torch::Tensor q_input,       
    torch::Tensor k_input,       
    torch::Tensor k_buf,         
    torch::Tensor k_loc,         
    int page_size,
    c10::optional<torch::Tensor> weights_in_opt = c10::nullopt,
    float q_scale_factor = 1.0f,  
    float k_scale_factor = 1.0f,  
    float eps = 1e-5f,        
    bool use_ue8m0 = false,
    bool is_e4m3 = true)      
{
    q_input = q_input.is_contiguous() ? q_input : q_input.contiguous();
    k_input = k_input.is_contiguous() ? k_input : k_input.contiguous();
    k_loc = k_loc.is_contiguous() ? k_loc : k_loc.contiguous();
    
    TORCH_CHECK(k_loc.scalar_type() == torch::kInt64, "loc tensor must be int64");
    TORCH_CHECK(q_input.scalar_type() == k_input.scalar_type(), "Q and K input must have the same precision");

    int const q_hidden_size = q_input.size(-1);
    int const k_hidden_size = k_input.size(-1);
    
   
    int64_t const num_tokens = k_loc.size(0); 
    int64_t const num_q_elements = q_input.numel() / q_hidden_size; 

 
    TORCH_CHECK(k_input.numel() / k_hidden_size == num_tokens, "当前内核的 K-Cache 写入逻辑仅支持单 Head 的 K 向量。");

    TORCH_CHECK(q_hidden_size <= 1024 && k_hidden_size <= 1024, "This fused kernel currently targets small chunks (<=1024 hidden_size).");
    TORCH_CHECK(q_hidden_size >= 16 && q_hidden_size % 16 == 0, "q_hidden_size must be multiple of 16");
    TORCH_CHECK(k_hidden_size >= 16 && k_hidden_size % 16 == 0, "k_hidden_size must be multiple of 16");

    bool has_weights = weights_in_opt.has_value() && weights_in_opt.value().defined();
    torch::Tensor weights_in, weights_out;

    if (has_weights) {
        weights_in = weights_in_opt.value().contiguous();
        weights_out = torch::empty_like(weights_in); 
    } else {
        weights_out = torch::empty({0}, q_input.options());
    }
    
    auto q_sizes = q_input.sizes().vec();
    q_sizes.back() = 1;

    auto q_out = torch::empty_like(q_input, q_input.options().dtype(torch::kFloat8_e4m3fn));
    auto q_scale_out = torch::empty(q_sizes, q_input.options().dtype(torch::kFloat32));

    int64_t s_offset_nbytes_in_page = page_size * k_hidden_size; 
    int64_t buf_numel_per_page = page_size * k_hidden_size + page_size * sizeof(float);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    int q_threads_per_group = q_hidden_size / 16;
    int k_threads_per_group = k_hidden_size / 16;
    
    constexpr int BLOCK_SIZE = 256; 
    
    int q_groups_per_block = BLOCK_SIZE / q_threads_per_group;
    int k_groups_per_block = BLOCK_SIZE / k_threads_per_group;
    
    int q_blocks = (num_q_elements + q_groups_per_block - 1) / q_groups_per_block;
    int k_blocks = (num_tokens + k_groups_per_block - 1) / k_groups_per_block;

    int total_blocks = q_blocks + k_blocks;

    VLLM_DISPATCH_FLOATING_TYPES(
        q_input.scalar_type(), "fused_qk_quant_weight_mul_and_store_index_k_cache", [&] {
            
            scalar_t* w_in_ptr = has_weights ? weights_in.data_ptr<scalar_t>() : nullptr;
            scalar_t* w_out_ptr = has_weights ? weights_out.data_ptr<scalar_t>() : nullptr;

            if (is_e4m3) {
                fused_qk_quant_weight_mul_and_store_index_k_cache<scalar_t, true>
                    <<<total_blocks, BLOCK_SIZE, 0, stream>>>(
                        q_input.data_ptr<scalar_t>(),
                        reinterpret_cast<uint8_t*>(q_out.data_ptr()),
                        q_scale_out.data_ptr<float>(),
                        w_in_ptr, w_out_ptr, 
                        q_scale_factor, k_scale_factor,  
                        q_hidden_size, q_threads_per_group, 
                        k_input.data_ptr<scalar_t>(),
                        reinterpret_cast<uint8_t*>(k_buf.data_ptr()),
                        k_loc.data_ptr<int64_t>(),
                        k_hidden_size, k_threads_per_group,
                        page_size, buf_numel_per_page, s_offset_nbytes_in_page,
                        
                        eps, use_ue8m0, num_q_elements, num_tokens, q_blocks
                    );
            } else {
                fused_qk_quant_weight_mul_and_store_index_k_cache<scalar_t, false>
                    <<<total_blocks, BLOCK_SIZE, 0, stream>>>(
                        q_input.data_ptr<scalar_t>(),
                        reinterpret_cast<uint8_t*>(q_out.data_ptr()),
                        q_scale_out.data_ptr<float>(),
                        w_in_ptr, w_out_ptr, 
                        q_scale_factor, k_scale_factor, 
                        q_hidden_size, q_threads_per_group, 
                        k_input.data_ptr<scalar_t>(),
                        reinterpret_cast<uint8_t*>(k_buf.data_ptr()),
                        k_loc.data_ptr<int64_t>(),
                        k_hidden_size, k_threads_per_group,
                        page_size, buf_numel_per_page, s_offset_nbytes_in_page,
                        
                        eps, use_ue8m0, num_q_elements, num_tokens, q_blocks
                    );
            }
        }
    );
    
    if (has_weights) {
        return std::make_tuple(q_out, q_scale_out, weights_out.unsqueeze(-1));
    } else {
        return std::make_tuple(q_out, q_scale_out, weights_out);
    }
}

} // namespace native
} // namespace at