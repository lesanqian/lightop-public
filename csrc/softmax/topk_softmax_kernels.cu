// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

constexpr float LOG2_E = 1.44269504089f; 
#define EXP_FUNC(x) __builtin_amdgcn_exp2f((x) * LOG2_E)

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)
 #endif

#ifndef USE_ROCM
    #include <cub/util_type.cuh>
    #include <cub/cub.cuh>
#else
    #include <hipcub/util_type.hpp>
    #include <hipcub/hipcub.hpp>
#endif
#define WARP_SIZE 64
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace at {
namespace native {

#if defined(__gfx946__)
// ============================================================================
// gfx946（少伯）TopK-Softmax 硬件原语（遵循 DTK 27.04 实测勘误约束）
//   1. 跨 16/32 lane 归约采用双 "+v" 内联汇编 v_permlane{16,32}_swap_b32，
//      绕开 LLVM 前端内置原型缺失第 2 路写回的问题，以 1-Pass VALU 替代
//      DS 管线的 ds_bpermute_b32（消除 lgkmcnt 等待）。
//   2. 有符号 BF16/FP16 打包极值：BF16 固定 op_sel_lo=0, op_sel_hi=7，
//      FP16 直接发射单条 v_pk_max_f16（保留符号位，用于 logits 行最大值）。
//   3. 2-way FP32 向量类型 ext_vector_type(2) 在 -O3 下自动生成 v_pk_*_f32。
// ============================================================================
using hcu_topk_v2f = float __attribute__((ext_vector_type(2)));
using hcu_topk_v2s = short __attribute__((ext_vector_type(2)));

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

__device__ __forceinline__ float hcu_permlane16_swap_fadd(float val) {
    float other = val;
    asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(val), "+v"(other));
    return val + other;
}

__device__ __forceinline__ float hcu_permlane32_swap_fadd(float val) {
    float other = val;
    asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(val), "+v"(other));
    return val + other;
}

__device__ __forceinline__ uint32_t hcu_permlane16_swap_u32_max(uint32_t val) {
    uint32_t other = val;
    asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(val), "+v"(other));
    return other > val ? other : val;
}

__device__ __forceinline__ uint32_t hcu_permlane32_swap_u32_max(uint32_t val) {
    uint32_t other = val;
    asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(val), "+v"(other));
    return other > val ? other : val;
}

__device__ __forceinline__ void hcu_permlane16_swap_argmax(float& val, int& idx) {
    float other_val = val;
    int other_idx = idx;
    asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(val), "+v"(other_val));
    asm volatile("v_permlane16_swap_b32 %0, %1" : "+v"(idx), "+v"(other_idx));
    const bool swap = (other_val > val) || (other_val == val && other_idx < idx);
    val = swap ? other_val : val;
    idx = swap ? other_idx : idx;
}

__device__ __forceinline__ void hcu_permlane32_swap_argmax(float& val, int& idx) {
    float other_val = val;
    int other_idx = idx;
    asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(val), "+v"(other_val));
    asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(idx), "+v"(other_idx));
    const bool swap = (other_val > val) || (other_val == val && other_idx < idx);
    val = swap ? other_val : val;
    idx = swap ? other_idx : idx;
}

__device__ __forceinline__ uint32_t hcu_pk_max_bf16_u32(uint32_t a_u32, uint32_t b_u32) {
    hcu_topk_v2s va = __builtin_bit_cast(hcu_topk_v2s, a_u32);
    hcu_topk_v2s vb = __builtin_bit_cast(hcu_topk_v2s, b_u32);
    hcu_topk_v2s vm = __builtin_hcu_pk_max_bf16(va, vb, 0, 7);
    return __builtin_bit_cast(uint32_t, vm);
}

__device__ __forceinline__ float hcu_bf16_pair_max_to_f32(uint32_t pair_u32) {
    const uint32_t lo_bits = (pair_u32 & 0xFFFFU) << 16;
    const uint32_t hi_bits = pair_u32 & 0xFFFF0000U;
    return fmaxf(__uint_as_float(lo_bits), __uint_as_float(hi_bits));
}

__device__ __forceinline__ uint32_t hcu_pk_max_f16_u32(uint32_t a_u32, uint32_t b_u32) {
    uint32_t res;
    asm("v_pk_max_f16 %0, %1, %2" : "=v"(res) : "v"(a_u32), "v"(b_u32));
    return res;
}

__device__ __forceinline__ float hcu_f16_pair_max_to_f32(uint32_t pair_u32) {
    c10::Half lo;
    c10::Half hi;
    lo.x = static_cast<uint16_t>(pair_u32 & 0xFFFFU);
    hi.x = static_cast<uint16_t>(pair_u32 >> 16);
    return fmaxf(static_cast<float>(lo), static_cast<float>(hi));
}

template <int VPT, typename input_t>
__device__ __forceinline__ float hcu_packed16_thread_max(const input_t (&temp_chunk)[VPT]) {
    static_assert(VPT >= 2 && (VPT % 2 == 0));
    const uint32_t* pairs = reinterpret_cast<const uint32_t*>(temp_chunk);
    uint32_t max_pair = pairs[0];
    if constexpr (std::is_same_v<input_t, c10::BFloat16>) {
#pragma unroll
        for (int i = 1; i < VPT / 2; ++i) {
            max_pair = hcu_pk_max_bf16_u32(max_pair, pairs[i]);
        }
        return hcu_bf16_pair_max_to_f32(max_pair);
    } else {
#pragma unroll
        for (int i = 1; i < VPT / 2; ++i) {
            max_pair = hcu_pk_max_f16_u32(max_pair, pairs[i]);
        }
        return hcu_f16_pair_max_to_f32(max_pair);
    }
}
#endif

template <int THREADS_PER_ROW>
__device__ __forceinline__ float topk_row_reduce_max(float val) {
#if defined(__gfx946__)
    if constexpr (THREADS_PER_ROW >= 64) {
        val = hcu_permlane32_swap_fmax(val);
    }
    if constexpr (THREADS_PER_ROW >= 32) {
        val = hcu_permlane16_swap_fmax(val);
    }
    constexpr int kStartMask = (THREADS_PER_ROW > 16) ? 8 : (THREADS_PER_ROW / 2);
#pragma unroll
    for (int mask = kStartMask; mask > 0; mask >>= 1) {
        val = fmaxf(val, VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, THREADS_PER_ROW));
    }
    return val;
#else
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        val = std::max(val, VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, THREADS_PER_ROW));
    }
    return val;
#endif
}

template <int THREADS_PER_ROW>
__device__ __forceinline__ float topk_row_reduce_sum(float val) {
#if defined(__gfx946__)
    if constexpr (THREADS_PER_ROW >= 64) {
        val = hcu_permlane32_swap_fadd(val);
    }
    if constexpr (THREADS_PER_ROW >= 32) {
        val = hcu_permlane16_swap_fadd(val);
    }
    constexpr int kStartMask = (THREADS_PER_ROW > 16) ? 8 : (THREADS_PER_ROW / 2);
#pragma unroll
    for (int mask = kStartMask; mask > 0; mask >>= 1) {
        val += VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, THREADS_PER_ROW);
    }
    return val;
#else
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        val += VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, THREADS_PER_ROW);
    }
    return val;
#endif
}

template <int THREADS_PER_ROW>
__device__ __forceinline__ void topk_row_reduce_argmax(float& val, int& idx) {
#if defined(__gfx946__)
    if constexpr (THREADS_PER_ROW >= 64) {
        hcu_permlane32_swap_argmax(val, idx);
    }
    if constexpr (THREADS_PER_ROW >= 32) {
        hcu_permlane16_swap_argmax(val, idx);
    }
    constexpr int kStartMask = (THREADS_PER_ROW > 16) ? 8 : (THREADS_PER_ROW / 2);
#pragma unroll
    for (int mask = kStartMask; mask > 0; mask >>= 1) {
        const float other_val = VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, THREADS_PER_ROW);
        const int other_idx = VLLM_SHFL_XOR_SYNC_WIDTH(idx, mask, THREADS_PER_ROW);
        const bool swap = (other_val > val) || (other_val == val && other_idx < idx);
        val = swap ? other_val : val;
        idx = swap ? other_idx : idx;
    }
#else
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        const float other_val = VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, THREADS_PER_ROW);
        const int other_idx = VLLM_SHFL_XOR_SYNC_WIDTH(idx, mask, THREADS_PER_ROW);
        const bool swap = (other_val > val) || (other_val == val && other_idx < idx);
        val = swap ? other_val : val;
        idx = swap ? other_idx : idx;
    }
#endif
}

template <int VPT>
__device__ __forceinline__ float topk_exp_and_sum_chunk(float (&row_chunk)[VPT], float thread_max) {
#if defined(__gfx946__)
    if constexpr (VPT >= 2 && (VPT % 2 == 0)) {
        const hcu_topk_v2f v_max = {thread_max, thread_max};
        const hcu_topk_v2f v_log2e = {LOG2_E, LOG2_E};
        hcu_topk_v2f v_sum = {0.0f, 0.0f};
#pragma unroll
        for (int ii = 0; ii < VPT; ii += 2) {
            const hcu_topk_v2f v_in = {row_chunk[ii], row_chunk[ii + 1]};
            const hcu_topk_v2f v_scaled = (v_in - v_max) * v_log2e;
            const float e0 = __builtin_amdgcn_exp2f(v_scaled[0]);
            const float e1 = __builtin_amdgcn_exp2f(v_scaled[1]);
            row_chunk[ii] = e0;
            row_chunk[ii + 1] = e1;
            const hcu_topk_v2f v_e = {e0, e1};
            v_sum = v_sum + v_e;
        }
        return v_sum[0] + v_sum[1];
    }
#endif
    float row_sum = 0.0f;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = EXP_FUNC(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }
    return row_sum;
}

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
    T data[N]; 
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <int TPB, typename input_t>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const input_t* input, const bool* finished, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int64_t thread_row_offset = static_cast<int64_t>(blockIdx.x) * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int64_t idx = thread_row_offset + ii;
        threadData = std::max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int64_t idx = thread_row_offset + ii;
        threadData += std::exp((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int64_t idx = thread_row_offset + ii;
        const float val = std::exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx] = val;
    }
}

template <int TPB, typename IndType>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int64_t thread_read_offset = static_cast<int64_t>(blockIdx.x) * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int64_t idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[static_cast<int64_t>(k) * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int64_t idx = static_cast<int64_t>(k) * block_row + k_idx;
            output[idx] = result_kvp.value;
            indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
            assert(indices[idx] >= 0);
            if (source_rows != nullptr) {
                source_rows[idx] = k_idx * num_rows + block_row;
            }
        }
        __syncthreads();
    }
}


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT; 
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG; 

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;  

    const input_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;

    input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)  
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }
    
    float row_chunk[VPT];
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = std::max(thread_max, row_chunk[ii]);
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = std::max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;   
    }

    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;  
     float topk_sum = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        float max_val = row_chunk[0]; 
        int expert = start_col; 
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }
       
        if (thread_group_idx == 0)
        {
            const bool node_uses_expert = expert >= start_expert && expert < end_expert; 
            const bool should_process_row = row_is_active && node_uses_expert; 
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            topk_sum  += max_val;  
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;  
            if (source_rows != nullptr) {
                source_rows[idx] = k_idx * num_rows + thread_row;  
            }
        }

        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;  
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;   

            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }
    if constexpr(Is_renormalize){
         if (thread_group_idx == 0 && row_is_active)
        {
            const float r_topk_sum = 1.0f / (topk_sum + 1e-8f);
            const int idx = k * thread_row;
            for (int k_idx = 0; k_idx < k; ++k_idx)
            {
                output[idx + k_idx] *= r_topk_sum;
            }
        }
    }
       
}

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk4GatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");


    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); 
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    

    if (thread_row >= num_rows)
    {
        return;
    }
    float max_val;
    int expert;
    const bool row_is_active = finished ? !finished[thread_row] : true;


    const input_t* thread_row_ptr = input + static_cast<int64_t>(thread_row) * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    alignas(sizeof(AccessType)) input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }
    
    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;
#pragma unroll
    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
        }
    }

    topk_row_reduce_argmax<THREADS_PER_ROW>(thread_max, thread_max_idx);

    float reciprocal_row_sum = 1.0f;
    if constexpr (!Is_renormalize) {
        float row_sum = topk_exp_and_sum_chunk<VPT>(row_chunk, thread_max);
        row_sum = topk_row_reduce_sum<THREADS_PER_ROW>(row_sum);
        reciprocal_row_sum = 1.f / row_sum;
    }

    float top_vals[4];
    IndType top_indices[4];

    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        int current_col_base = first_elt_read_by_thread; \
       for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        topk_row_reduce_argmax<THREADS_PER_ROW>(k_val, k_expert); \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert != -1) { \
            int expert_in_thread_col = (k_expert / ELTS_PER_LDG) % THREADS_PER_ROW; \
            if (thread_group_idx == expert_in_thread_col) { \
                int ldg_idx = k_expert / (THREADS_PER_ROW * ELTS_PER_LDG); \
                int elt_idx = k_expert % ELTS_PER_LDG; \
                row_chunk[ldg_idx * ELTS_PER_LDG + elt_idx] = -FLT_MAX; \
            } \
        }

    if (thread_group_idx == 0) {
        top_vals[0] = Is_renormalize ? thread_max : 1.0f;
        top_indices[0] = thread_max_idx;
    }
    ERASE_MAX(thread_max_idx);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[1] = max_val; top_indices[1] = expert; }
    ERASE_MAX(expert);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[2] = max_val; top_indices[2] = expert; }
    ERASE_MAX(expert);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[3] = max_val; top_indices[3] = expert; }

    #undef FIND_AND_REDUCE_MAX
    #undef ERASE_MAX

    if (thread_group_idx == 0 && row_is_active)
    {
        float final_scale = reciprocal_row_sum;
        if constexpr(Is_renormalize){
            top_vals[0] = 1.0f;
            top_vals[1] = EXP_FUNC(top_vals[1] - thread_max);
            top_vals[2] = EXP_FUNC(top_vals[2] - thread_max);
            top_vals[3] = EXP_FUNC(top_vals[3] - thread_max);
            const float topk_sum = top_vals[0] + top_vals[1] + top_vals[2] + top_vals[3];
            final_scale = 1.0f / topk_sum;
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            top_vals[i] *= final_scale;
            const bool node_uses_expert = top_indices[i] >= start_expert && top_indices[i] < end_expert;
            top_indices[i] = node_uses_expert ? (top_indices[i] - start_expert) : NUM_EXPERTS;
        }

        const int64_t base_idx = static_cast<int64_t>(4) * thread_row;
        *reinterpret_cast<float4*>(&output[base_idx]) = *reinterpret_cast<float4*>(&top_vals[0]);
      
        if (source_rows != nullptr) {
            int source_rows_temp[4];
            source_rows_temp[0] = 0 * num_rows + thread_row;
            source_rows_temp[1] = 1 * num_rows + thread_row;
            source_rows_temp[2] = 2 * num_rows + thread_row;
            source_rows_temp[3] = 3 * num_rows + thread_row;
            *reinterpret_cast<int4*>(&source_rows[base_idx]) = *reinterpret_cast<int4*>(&source_rows_temp[0]);
        }

        for (int i = 0; i < 4; ++i) {
            indices[base_idx + i] = top_indices[i];
        }
    }
}


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topk2GatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
    int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
    {
        return;
    }

    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + static_cast<int64_t>(thread_row) * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    
    alignas(sizeof(AccessType)) input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float thread_max = -FLT_MAX;
#if defined(__gfx946__)
    if constexpr ((std::is_same_v<input_t, c10::BFloat16> || std::is_same_v<input_t, c10::Half>) && VPT >= 2) {
        thread_max = hcu_packed16_thread_max<VPT, input_t>(temp_chunk);
    }
#endif

    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

#if defined(__gfx946__)
    if constexpr (!((std::is_same_v<input_t, c10::BFloat16> || std::is_same_v<input_t, c10::Half>) && VPT >= 2))
#endif
    {
#pragma unroll
        for(int i = 0; i < VPT; ++i) thread_max = std::max(thread_max, row_chunk[i]);
    }

    thread_max = topk_row_reduce_max<THREADS_PER_ROW>(thread_max);

    float reciprocal_row_sum = 1.0f;
    if constexpr (!Is_renormalize) {
        float row_sum = topk_exp_and_sum_chunk<VPT>(row_chunk, thread_max);
        row_sum = topk_row_reduce_sum<THREADS_PER_ROW>(row_sum);
        reciprocal_row_sum = 1.f / row_sum;
    }

    float top_vals[2];
    IndType top_indices[2];
    
    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        int current_col_base = first_elt_read_by_thread; \
       for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        topk_row_reduce_argmax<THREADS_PER_ROW>(k_val, k_expert); \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert != -1) { \
            int expert_in_thread_col = (k_expert / ELTS_PER_LDG) % THREADS_PER_ROW; \
            if (thread_group_idx == expert_in_thread_col) { \
                int ldg_idx = k_expert / (THREADS_PER_ROW * ELTS_PER_LDG); \
                int elt_idx = k_expert % ELTS_PER_LDG; \
                row_chunk[ldg_idx * ELTS_PER_LDG + elt_idx] = -FLT_MAX; \
            } \
        }

    float max_val;
    int expert;

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[0] = max_val; top_indices[0] = expert; }
    ERASE_MAX(expert);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[1] = max_val; top_indices[1] = expert; }

    #undef FIND_AND_REDUCE_MAX
    #undef ERASE_MAX

    if (thread_group_idx == 0 && row_is_active)
    {
        float final_scale = reciprocal_row_sum;
        if constexpr(Is_renormalize){
            top_vals[0] = EXP_FUNC(top_vals[0] - thread_max);
            top_vals[1] = EXP_FUNC(top_vals[1] - thread_max);
            const float topk_sum = top_vals[0] + top_vals[1];
            final_scale = 1.0f / topk_sum;
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
            top_vals[i] *= final_scale;
            const bool node_uses_expert = top_indices[i] >= start_expert && top_indices[i] < end_expert;
            top_indices[i] = node_uses_expert ? (top_indices[i] - start_expert) : NUM_EXPERTS;
        }

        const int64_t base_idx = static_cast<int64_t>(2) * thread_row;
        *reinterpret_cast<float2*>(&output[base_idx]) = *reinterpret_cast<float2*>(&top_vals[0]);
        if (source_rows != nullptr) {
            int source_rows_temp[2];
            source_rows_temp[0] = 0 * num_rows + thread_row;
            source_rows_temp[1] = 1 * num_rows + thread_row;
            *reinterpret_cast<int2*>(&source_rows[base_idx]) = *reinterpret_cast<int2*>(&source_rows_temp[0]);
        }

        if constexpr (std::is_same_v<IndType, int>) {
            *reinterpret_cast<int2*>(&indices[base_idx]) = *reinterpret_cast<int2*>(&top_indices[0]);
        } else { 
            for (int i = 0; i < 2; ++i) {
                indices[base_idx + i] = top_indices[i];
            }
        }
    }
}

template <typename input_t>
__device__ __forceinline__ uint32_t topk_16bit_key(
    uint16_t raw_bits, int expert) {
    // Canonicalize -0.0 (0x8000) to +0.0 (0x0000) so equal zeros keep smaller-expert tie-breaking.
    if (raw_bits == 0x8000u) {
        raw_bits = 0x0000u;
    }
    const uint32_t ordered =
        (raw_bits & 0x8000u)
            ? ((~static_cast<uint32_t>(raw_bits)) & 0xffffu)
            : (static_cast<uint32_t>(raw_bits) ^ 0x8000u);
    return (ordered << 16) | (0xffffu - static_cast<uint32_t>(expert));
}

template <typename input_t>
__device__ __forceinline__ float topk_16bit_key_value(uint32_t key) {
    const uint16_t ordered = static_cast<uint16_t>(key >> 16);
    const uint16_t raw_bits =
        (ordered & 0x8000u)
            ? static_cast<uint16_t>(ordered ^ 0x8000u)
            : static_cast<uint16_t>(~ordered);
    if constexpr (std::is_same_v<input_t, c10::BFloat16>) {
        return __uint_as_float(static_cast<uint32_t>(raw_bits) << 16);
    } else {
        c10::Half value;
        value.x = raw_bits;
        return static_cast<float>(value);
    }
}

template <int DppCtrl>
__device__ __forceinline__ uint32_t topk_mov_dpp(uint32_t value) {
    return static_cast<uint32_t>(
        __builtin_amdgcn_mov_dpp(
            value, DppCtrl, 0xf, 0xf, true));
}

template <int kThreadsPerRow>
__device__ __forceinline__ uint32_t topk_row_reduce_key_max(uint32_t winner_key) {
    static_assert(kThreadsPerRow >= 8 && kThreadsPerRow <= WARP_SIZE);
    static_assert((kThreadsPerRow & (kThreadsPerRow - 1)) == 0);
#if defined(__gfx946__) && defined(LIGHTOP_GFX946_TOPK_DISABLE_DPP)
    if constexpr (kThreadsPerRow >= 64) {
        winner_key = hcu_permlane32_swap_u32_max(winner_key);
    }
    if constexpr (kThreadsPerRow >= 32) {
        winner_key = hcu_permlane16_swap_u32_max(winner_key);
    }
    constexpr int kStartMask = (kThreadsPerRow > 16) ? 8 : (kThreadsPerRow / 2);
#pragma unroll
    for (int mask = kStartMask; mask > 0; mask >>= 1) {
        const uint32_t other_key = VLLM_SHFL_XOR_SYNC_WIDTH(winner_key, mask, kThreadsPerRow);
        winner_key = other_key > winner_key ? other_key : winner_key;
    }
    return winner_key;
#else
    uint32_t other_key = topk_mov_dpp<0xb1>(winner_key);
    winner_key = other_key > winner_key ? other_key : winner_key;
    other_key = topk_mov_dpp<0x4e>(winner_key);
    winner_key = other_key > winner_key ? other_key : winner_key;
    other_key = VLLM_SHFL_XOR_SYNC_WIDTH(winner_key, 4, kThreadsPerRow);
    winner_key = other_key > winner_key ? other_key : winner_key;
    if constexpr (kThreadsPerRow >= 16) {
        other_key = topk_mov_dpp<0x128>(winner_key);
        winner_key = other_key > winner_key ? other_key : winner_key;
    }
    if constexpr (kThreadsPerRow >= 32) {
#if defined(__gfx946__)
        winner_key = hcu_permlane16_swap_u32_max(winner_key);
#else
        other_key = VLLM_SHFL_XOR_SYNC_WIDTH(winner_key, 16, kThreadsPerRow);
        winner_key = other_key > winner_key ? other_key : winner_key;
#endif
    }
    if constexpr (kThreadsPerRow >= 64) {
#if defined(__gfx946__)
        winner_key = hcu_permlane32_swap_u32_max(winner_key);
#else
        other_key = VLLM_SHFL_XOR_SYNC_WIDTH(winner_key, 32, kThreadsPerRow);
        winner_key = other_key > winner_key ? other_key : winner_key;
#endif
    }
    return winner_key;
#endif
}

// 1-Pass 16-bit packed TopK-8 (Is_renormalize=true) supporting both FP16 and BF16,
// all IndType, and kExperts in {128, 256, 512}. On gfx946:
//   - Avoids expanding temp_chunk into float[VPT] to fit 512 KB VGPR/CU budget.
//   - Uses 1-Pass VALU v_permlane{16,32}_swap_b32 for kThreadsPerRow in {32, 64}.
template <int kExperts, int kThreadsPerRow, typename IndType, typename input_t>
__launch_bounds__(WARP_SIZE) __global__
void topk8GatingSoftmax_packed16(
    const input_t* input,
    const bool* finished,
    float* output,
    const int num_rows,
    IndType* indices,
    int* source_rows,
    const int start_expert,
    const int end_expert)
{
    static_assert(std::is_same_v<input_t, c10::Half> || std::is_same_v<input_t, c10::BFloat16>);
    constexpr int kTopK = 8;
    constexpr int kValuesPerThread = kExperts / kThreadsPerRow;
    constexpr int kRowsPerWave = WARP_SIZE / kThreadsPerRow;
    static_assert(kExperts % kThreadsPerRow == 0);
    static_assert(kThreadsPerRow >= kTopK && kThreadsPerRow <= WARP_SIZE);
    static_assert(WARP_SIZE % kThreadsPerRow == 0);

    const int row_in_wave = threadIdx.x / kThreadsPerRow;
    const int lane_in_row = threadIdx.x % kThreadsPerRow;
    const int row = blockIdx.x * kRowsPerWave + row_in_wave;
    if (row >= num_rows || (finished != nullptr && finished[row])) {
        return;
    }

    const int first_expert = lane_in_row * kValuesPerThread;
    const uint16_t* row_ptr = reinterpret_cast<const uint16_t*>(
        input + static_cast<int64_t>(row) * kExperts);
    using LoadType = AlignedArray<uint16_t, kValuesPerThread>;
    alignas(sizeof(LoadType)) uint16_t packed_values[kValuesPerThread];
    *reinterpret_cast<LoadType*>(packed_values) =
        *reinterpret_cast<const LoadType*>(row_ptr + first_expert);

    uint32_t keys[kValuesPerThread];
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        keys[i] = topk_16bit_key<input_t>(
            packed_values[i], first_expert + i);
    }

    uint32_t stored_key = 0;
    float row_max = -FLT_MAX;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        uint32_t winner_key = keys[0];
#pragma unroll
        for (int i = 1; i < kValuesPerThread; ++i) {
            winner_key = keys[i] > winner_key ? keys[i] : winner_key;
        }
        winner_key = topk_row_reduce_key_max<kThreadsPerRow>(winner_key);

        if (rank == 0) {
            row_max = topk_16bit_key_value<input_t>(winner_key);
        }
        if (lane_in_row == rank) {
            stored_key = winner_key;
        }
        if (rank + 1 < kTopK) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                if (keys[i] == winner_key) {
                    keys[i] = 0;
                }
            }
        }
    }

    float value_exp = 0.0f;
    if (lane_in_row < kTopK) {
        value_exp =
            EXP_FUNC(topk_16bit_key_value<input_t>(stored_key) - row_max);
    }
    float topk_sum = value_exp;
#pragma unroll
    for (int mask = kTopK / 2; mask > 0; mask >>= 1) {
        topk_sum +=
            VLLM_SHFL_XOR_SYNC_WIDTH(topk_sum, mask, kThreadsPerRow);
    }

    if (lane_in_row < kTopK) {
        int top_expert =
            0xffff - static_cast<int>(stored_key & 0xffffu);
        const bool node_uses_expert =
            top_expert >= start_expert && top_expert < end_expert;
        top_expert =
            node_uses_expert ? top_expert - start_expert : kExperts;
        const int64_t output_offset =
            static_cast<int64_t>(row) * kTopK + lane_in_row;
        output[output_offset] =
            value_exp * (1.0f / (topk_sum + 1e-6f));
        indices[output_offset] = static_cast<IndType>(top_expert);
        if (source_rows != nullptr) {
            source_rows[output_offset] =
                lane_in_row * num_rows + row;
        }
    }
}

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk8GatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); 
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    
    if (thread_row >= num_rows)
    {
        return;
    }
    float max_val;
    int expert;
 
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + static_cast<int64_t>(thread_row) * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    alignas(sizeof(AccessType)) input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;
#pragma unroll
    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
        }
    }

    topk_row_reduce_argmax<THREADS_PER_ROW>(thread_max, thread_max_idx);

 
    float reciprocal_row_sum = 0.0f;

   
    if constexpr (!Is_renormalize) {
        float row_sum = topk_exp_and_sum_chunk<VPT>(row_chunk, thread_max);
        row_sum = topk_row_reduce_sum<THREADS_PER_ROW>(row_sum);
        reciprocal_row_sum = 1.f / (row_sum + 1e-6f);
    }

    
    float top_vals = 0.f;
    IndType top_indices = IndType(0);

    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        _Pragma("unroll") \
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            _Pragma("unroll") \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = first_elt_read_by_thread + ldg * THREADS_PER_ROW * ELTS_PER_LDG + ii; \
                } \
            } \
        } \
        topk_row_reduce_argmax<THREADS_PER_ROW>(k_val, k_expert); \
    } while(0)

    #define ERASE_MAX(k_expert_to_erase) \
    do { \
        _Pragma("unroll") \
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            _Pragma("unroll") \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                int col_idx = first_elt_read_by_thread + ldg * THREADS_PER_ROW * ELTS_PER_LDG + ii; \
                if (col_idx == k_expert_to_erase) { \
                    row_chunk[ldg * ELTS_PER_LDG + ii] = -FLT_MAX; \
                } \
            } \
        } \
    } while(0)

    {
        constexpr int i = 0;
        if (thread_group_idx == i) { 
            if constexpr (Is_renormalize) {
                top_vals = thread_max; 
            } else {
                top_vals = 1.0f;       
            }
            top_indices = thread_max_idx; 
        }
        ERASE_MAX(thread_max_idx); 
    }
    {
        constexpr int i = 1;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 2;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 3;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 4;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 5;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 6;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 7;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
    }

    #undef FIND_AND_REDUCE_MAX
    #undef ERASE_MAX

    if(row_is_active){
        float final_scale = 1.0f; 

        if constexpr(Is_renormalize){
            float val_exp = 0.0f;
            if (thread_group_idx < 8) {
                val_exp = EXP_FUNC(top_vals - thread_max);
                top_vals = val_exp; 
            }
            float topk_sum = val_exp;
    #pragma unroll
            for(int i = 4; i > 0; i >>= 1) {
                topk_sum += VLLM_SHFL_XOR_SYNC_WIDTH(topk_sum,i,THREADS_PER_ROW);
            }
            final_scale = 1.0f / (topk_sum + 1e-6f);
        } else {
             final_scale = reciprocal_row_sum;
        }

        if(thread_group_idx < 8){
            top_vals *= final_scale;
            
            const bool node_uses_expert = top_indices >= start_expert && top_indices  < end_expert;
            top_indices = node_uses_expert ? (top_indices - start_expert) : NUM_EXPERTS;
            const int64_t base_idx = static_cast<int64_t>(8) * thread_row + thread_group_idx;
            indices[base_idx] = top_indices;
            output[base_idx] = top_vals;
            
            if (source_rows != nullptr) {
                source_rows[base_idx] = thread_group_idx * num_rows + thread_row;
            }
        }
    }
}


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topk8GatingSoftmax_no_align(
    const input_t* input, 
    const bool* finished, 
    float* output,   
    const int num_rows, 
    IndType* indices,
    int* source_rows, 
    const int start_expert, 
    const int end_expert, 
    const int model_num_experts) 
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); 
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    
    if (thread_row >= num_rows)
    {
        return;
    }
    
    float max_val;
    int expert;
    int my_search_expert_idx = -1; 
    int my_search_reg_idx = -1;   

    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + static_cast<int64_t>(thread_row) * model_num_experts;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    float row_chunk[VPT];
    
    #pragma unroll
    for (int i=0; i<VPT; ++i) row_chunk[i] = -FLT_MAX;

    #pragma unroll
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
        int offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
        int col_idx = thread_group_idx * ELTS_PER_LDG + offset;
        if (col_idx < model_num_experts) {
            #pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                if (col_idx + ii < model_num_experts) {
                    row_chunk[ldg * ELTS_PER_LDG + ii] = static_cast<float>(thread_read_ptr[offset + ii]);
                }
            }
        }
    }

    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;
    
    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
            
            my_search_reg_idx = i;
            my_search_expert_idx = thread_max_idx;
        }
    }

    topk_row_reduce_argmax<THREADS_PER_ROW>(thread_max, thread_max_idx);

    float reciprocal_row_sum = 0.0f; 

    if constexpr(!Is_renormalize){
        float row_sum = topk_exp_and_sum_chunk<VPT>(row_chunk, thread_max);
        row_sum = topk_row_reduce_sum<THREADS_PER_ROW>(row_sum);
        reciprocal_row_sum = 1.f / (row_sum + 1e-6f); 
    }
   
    float top_vals = 0.f; 
    IndType top_indices = IndType(0);

    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        my_search_reg_idx = -1; \
        int current_col_base = first_elt_read_by_thread; \
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                    my_search_reg_idx = ldg * ELTS_PER_LDG + ii; \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        my_search_expert_idx = k_expert; \
        topk_row_reduce_argmax<THREADS_PER_ROW>(k_val, k_expert); \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert == my_search_expert_idx) { \
            row_chunk[my_search_reg_idx] = -FLT_MAX; \
        }

    {
        constexpr int i = 0;
        if (thread_group_idx == i) { 
            if constexpr (Is_renormalize) {
                top_vals = thread_max; 
            } else {
                top_vals = 1.0f;       
            }
            top_indices = thread_max_idx; 
        }
        ERASE_MAX(thread_max_idx); 
    }
    {
        constexpr int i = 1;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 2;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 3;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 4;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 5;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 6;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 7;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
    }

    #undef FIND_AND_REDUCE_MAX
    #undef ERASE_MAX

    if(row_is_active){
        float final_scale = 1.0f;
        
        if constexpr(Is_renormalize){
            // 此时 top_vals 还是原始 Logits，需要转为 Exp 值
            float val_exp = 0.0f;
            if (thread_group_idx < 8) {
                val_exp = EXP_FUNC(top_vals - thread_max);
                top_vals = val_exp; 
            }
            
            // 计算局部 TopK Sum
            float topk_sum = val_exp;
            #pragma unroll
            for(int i = 4; i > 0; i >>= 1) {
                topk_sum += VLLM_SHFL_XOR_SYNC_WIDTH(topk_sum, i, THREADS_PER_ROW);
            }
            final_scale = 1.0f / (topk_sum + 1e-6f); 
        } else {
            final_scale = reciprocal_row_sum;
        }

        if(thread_group_idx < 8){
            top_vals *= final_scale;

            const bool node_uses_expert = top_indices >= start_expert && top_indices < model_num_experts;
            top_indices = node_uses_expert ? (top_indices - start_expert) : model_num_experts;

            const int64_t base_idx = static_cast<int64_t>(8) * thread_row + thread_group_idx;
            
            indices[base_idx] = top_indices;
            output[base_idx] = top_vals; 
            
            if (source_rows != nullptr) {
                source_rows[base_idx] = thread_group_idx * num_rows + thread_row;
            }
        }
    }
}

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topkGatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert, const int topk)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); // [修改点]
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    
    if (thread_row >= num_rows)
    {
        return;
    }
    float max_val;
    int expert;
 
    int my_search_expert_idx = -1; 
    int my_search_reg_idx = -1;
    
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + static_cast<int64_t>(thread_row) * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    alignas(sizeof(AccessType)) input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;
#pragma unroll
    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
        
            my_search_reg_idx = i;
            my_search_expert_idx = thread_max_idx;
        }
    }

    topk_row_reduce_argmax<THREADS_PER_ROW>(thread_max, thread_max_idx);

    float reciprocal_row_sum = 0.0f;

    if constexpr (!Is_renormalize) {
        float row_sum = topk_exp_and_sum_chunk<VPT>(row_chunk, thread_max);
        row_sum = topk_row_reduce_sum<THREADS_PER_ROW>(row_sum);
        reciprocal_row_sum = 1.f / (row_sum + 1e-6f);
    }

    
    float top_vals = 0.f;
    IndType top_indices = IndType(0);

    
    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        my_search_reg_idx = -1;  \
        int current_col_base = first_elt_read_by_thread; \
       for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                    my_search_reg_idx = ldg * ELTS_PER_LDG + ii;  \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        my_search_expert_idx = k_expert; \
        topk_row_reduce_argmax<THREADS_PER_ROW>(k_val, k_expert); \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert == my_search_expert_idx) { \
            row_chunk[my_search_reg_idx] = -FLT_MAX; \
        }

    //top1
    {
       
        if (thread_group_idx == 0) {
            if constexpr (Is_renormalize) {
                top_vals = thread_max; 
            } else {
                top_vals = 1.0f;       
            }
            top_indices = thread_max_idx;
        }
   
        ERASE_MAX(thread_max_idx);
    }

    for (int k = 1; k < topk; ++k) {
        float current_max_val;
        int current_expert;
        FIND_AND_REDUCE_MAX(current_max_val, current_expert);
        if (thread_group_idx == k) {
    
            top_vals = current_max_val;
            top_indices = current_expert;
        }

        ERASE_MAX(current_expert);
    }

    #undef FIND_AND_REDUCE_MAX
    #undef ERASE_MAX

    if(row_is_active){
        float final_scale = 1.0f; 

        if constexpr(Is_renormalize){
            float val_exp = 0.0f;
            if (thread_group_idx < topk) {
                val_exp = EXP_FUNC(top_vals - thread_max);
                top_vals = val_exp; 
            }
            float topk_sum = topk_row_reduce_sum<THREADS_PER_ROW>(val_exp);
            final_scale = 1.0f / (topk_sum + 1e-6f);
        } else {
             final_scale = reciprocal_row_sum;
        }

        if(thread_group_idx < topk){
            top_vals *= final_scale;
            const bool node_uses_expert = top_indices >= start_expert && top_indices  < end_expert;
            top_indices = node_uses_expert ? (top_indices - start_expert) : NUM_EXPERTS;
            const int64_t base_idx = static_cast<int64_t>(topk) * thread_row + thread_group_idx;
            indices[base_idx] = top_indices;
            output[base_idx] = top_vals;
            
            if (source_rows != nullptr) {
                source_rows[base_idx] = thread_group_idx * num_rows + thread_row;
            }
        }
    }
}
namespace detail
{
template <int EXPERTS, int BYTES_PER_LDG, typename input_t>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t);  
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");

    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;  
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT; 
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, typename IndType, bool Is_renormalize, typename input_t>
void topkGatingSoftmaxLauncherHelper(const input_t* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream,const int num_experts)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 32; 
    static constexpr int TARGET_THREADS_PER_ROW = (EXPERTS >= 8) ? 8 : EXPERTS;
    static constexpr int MAX_BYTES_FOR_THREADS = (EXPERTS * sizeof(input_t)) / TARGET_THREADS_PER_ROW;
    static constexpr int BYTES_PER_LDG = MIN((int)MAX_BYTES_PER_LDG, MAX_BYTES_FOR_THREADS);
    
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, input_t>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB / 1;
    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
 

    if(k == 4){
     topk4GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
     
    }else if(k == 2){
        topk2GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
    }else if(k == 8){
        if(num_experts == 160){
            topk8GatingSoftmax_no_align<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert, num_experts);
        }else if constexpr (
            (EXPERTS == 128 || EXPERTS == 256 || EXPERTS == 512) &&
            Is_renormalize &&
            (std::is_same_v<input_t, c10::Half> || std::is_same_v<input_t, c10::BFloat16>)) {
            const auto* props = at::cuda::getCurrentDeviceProperties();
            const bool is_gfx946 =
                props != nullptr && std::strncmp(props->gcnArchName, "gfx946", 6) == 0;
            // On gfx946 (192 CUs + 1-pass VALU v_permlane{16,32}_swap_b32), keep wider
            // threads-per-row up to 1024 tokens to halve VGPRs and increase CU occupancy.
            const int wide_row_threshold = is_gfx946 ? 1024 : 512;
            if constexpr (EXPERTS == 512) {
                if (num_rows <= wide_row_threshold) {
                    topk8GatingSoftmax_packed16<512, 64, IndType, input_t><<<
                        num_rows, WARP_SIZE, 0, stream>>>(
                        input, finished, output, num_rows, indices, source_row,
                        start_expert, end_expert);
                } else {
                    topk8GatingSoftmax_packed16<512, 32, IndType, input_t><<<
                        (num_rows + 1) / 2, WARP_SIZE, 0, stream>>>(
                        input, finished, output, num_rows, indices, source_row,
                        start_expert, end_expert);
                }
            } else {
                if (num_rows <= wide_row_threshold) {
                    topk8GatingSoftmax_packed16<EXPERTS, 32, IndType, input_t><<<
                        (num_rows + 1) / 2, WARP_SIZE, 0, stream>>>(
                        input, finished, output, num_rows, indices, source_row,
                        start_expert, end_expert);
                } else {
                    topk8GatingSoftmax_packed16<EXPERTS, 16, IndType, input_t><<<
                        (num_rows + 3) / 4, WARP_SIZE, 0, stream>>>(
                        input, finished, output, num_rows, indices, source_row,
                        start_expert, end_expert);
                }
            }
        }else{
            topk8GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
        }
    }
    else{
        topkGatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert,k);
    }
}

template <typename IndType, typename input_t>
void topkGatingSoftmaxKernelLauncher(
    const input_t* gating_output,
    float* topk_weights,
    IndType* topk_indicies,
    int* token_expert_indices,
    float* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    cudaStream_t stream,
    bool Is_renormalize) { 

    auto launch_kernel = [&](auto is_renormalize_flag) {
        static constexpr bool Is_renormalize = decltype(is_renormalize_flag)::value;
        static constexpr int WARPS_PER_TB = 4;
        #define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB) \
            topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, IndType, Is_renormalize, input_t>( \
                gating_output, nullptr, topk_weights, topk_indicies, \
                token_expert_indices, num_tokens, topk, 0, num_experts, \
                stream,num_experts);

        switch (num_experts) {
            case 1: LAUNCH_SOFTMAX(1, WARPS_PER_TB); break;
            case 2: LAUNCH_SOFTMAX(2, WARPS_PER_TB); break;
            case 4: LAUNCH_SOFTMAX(4, WARPS_PER_TB); break;
            case 8: LAUNCH_SOFTMAX(8, WARPS_PER_TB); break;
            case 16: LAUNCH_SOFTMAX(16, WARPS_PER_TB); break;
            case 32: LAUNCH_SOFTMAX(32, WARPS_PER_TB); break;
            /* 字节 */
            case 64:
                if(num_tokens < 16384) { LAUNCH_SOFTMAX(64, WARPS_PER_TB); }
                else { LAUNCH_SOFTMAX(64, 1); }
                break;
            /* qwen 30b*/
            case 128: {
                LAUNCH_SOFTMAX(128, 2); 
                break;
            }
            case 160:                         //160使用256的模板
            case 256: LAUNCH_SOFTMAX(256, 1); break;
            case 512: LAUNCH_SOFTMAX(512, 2); break;
            default: {
            TORCH_CHECK(softmax_workspace != nullptr,
                "softmax_workspace must be provided for num_experts that are not a power of 2.");
            static constexpr int TPB = 256;
            moeSoftmax<TPB, input_t><<<num_tokens, TPB, 0, stream>>>(
                gating_output, nullptr, softmax_workspace, num_experts);
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                softmax_workspace, nullptr, topk_weights, topk_indicies, token_expert_indices,
                num_experts, topk, 0, num_experts);
            }
        }
        #undef LAUNCH_SOFTMAX 
    };
    if (Is_renormalize) {
        launch_kernel(std::true_type{});
    } else {
        launch_kernel(std::false_type{});
    }
}

void topk_softmax(
    torch::Tensor& topk_weights,                // [num_tokens, topk]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::optional<torch::Tensor>& token_expert_indices, // [num_tokens, topk]
    torch::Tensor& gating_output,
    bool Is_renormalize = false)               // [num_tokens, num_experts]
{

    TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float, 
                "topk_weights must be Float32 (float)");
    TORCH_CHECK(gating_output.scalar_type() == at::ScalarType::Float || 
                gating_output.scalar_type() == at::ScalarType::Half ||
                gating_output.scalar_type() == at::ScalarType::BFloat16,
                "gating_output must be Float32, Float16 or BFloat16");

    const int num_experts = gating_output.size(-1);
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    const int64_t num_tokens_64 = gating_output.numel() / num_experts;
    TORCH_CHECK(num_tokens_64 <= std::numeric_limits<int>::max(),
                "num_tokens exceeds int32 range");
    const int num_tokens = static_cast<int>(num_tokens_64);
    const int topk = topk_weights.size(-1);
    if (num_tokens == 0) {
        return;
    }

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = (!is_pow_2 && num_experts != 160) || num_experts > 512;
    const int64_t workspace_size =
        needs_workspace ? (num_tokens_64 * static_cast<int64_t>(num_experts)) : 0;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor softmax_workspace;
    float* softmax_workspace_ptr = nullptr;
    if (needs_workspace) {
        softmax_workspace = torch::empty({workspace_size}, gating_output.options().dtype(at::kFloat));
        softmax_workspace_ptr = softmax_workspace.data_ptr<float>();
    }
   
    int* token_expert_indices_ptr = nullptr;
    if (token_expert_indices.has_value() && token_expert_indices.value().defined()) {
        token_expert_indices_ptr = token_expert_indices.value().data_ptr<int>();
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, gating_output.scalar_type(), "topk_softmax_cuda", [&] {
        if(topk_indices.scalar_type() == at::ScalarType::Int)
        {
           topkGatingSoftmaxKernelLauncher<int, scalar_t>(
                gating_output.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                topk_indices.data_ptr<int>(),
                token_expert_indices_ptr, 
                softmax_workspace_ptr,
                num_tokens,
                num_experts,
                topk,
                stream,
            Is_renormalize);
        }
        else if (topk_indices.scalar_type() == at::ScalarType::UInt32)
        {
           topkGatingSoftmaxKernelLauncher<uint32_t, scalar_t>(
                gating_output.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                topk_indices.data_ptr<uint32_t>(),
                token_expert_indices_ptr, 
                softmax_workspace_ptr,
                num_tokens,
                num_experts,
                topk,
                stream,
            Is_renormalize);
        }
        else {
            TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Long,
                        "topk_indices must be Int32, UInt32, or Int64");
           topkGatingSoftmaxKernelLauncher<int64_t, scalar_t>(
                gating_output.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                topk_indices.data_ptr<int64_t>(),
                token_expert_indices_ptr, 
                softmax_workspace_ptr,
                num_tokens,
                num_experts,
                topk,
                stream,
            Is_renormalize);
        }
    });
}

} // namespace native
} // namespace at
