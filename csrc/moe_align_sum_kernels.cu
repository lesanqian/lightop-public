// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cub/cub.cuh>
#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <cub/block/block_scan.cuh>
#include <cstdint>
#include <cstring>
#include <type_traits>
#define WARP_SIZE 1

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
template <typename T, int N> struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

namespace at {
namespace native {

// On gfx946 (128 KB LDS / CU), replicate the per-block expert histogram across
// multiple bank-padded copies (stride = smem_size + 1) to reduce LDS atomicAdd
// contention under skewed MoE routing during both count and scatter passes.
// Non-gfx946 architectures keep a single copy to stay within 64 KB LDS budgets.
template <int SMEM_SIZE>
__device__ __host__ constexpr int hcu_align_hist_copies() {
#if defined(__gfx946__)
  return (SMEM_SIZE <= 256) ? 8 : 4;
#else
  return 1;
#endif
}

__device__ __forceinline__ void fill_sorted_token_ids_vec(
    int32_t* __restrict__ dst,
    int64_t len,
    int32_t fill_val,
    size_t tid,
    size_t stride) {
  if (len <= 0) {
    return;
  }
  if ((reinterpret_cast<uintptr_t>(dst) & 0xFU) == 0) {
    const int64_t vec_len = len >> 2;
    int4* __restrict__ dst_vec = reinterpret_cast<int4*>(dst);
    const int4 v = make_int4(fill_val, fill_val, fill_val, fill_val);
    for (size_t i = tid; i < static_cast<size_t>(vec_len); i += stride) {
      dst_vec[i] = v;
    }
    const int64_t tail_base = vec_len << 2;
    for (int64_t i = tail_base + static_cast<int64_t>(tid); i < len;
         i += static_cast<int64_t>(stride)) {
      dst[i] = fill_val;
    }
  } else {
    for (size_t i = tid; i < static_cast<size_t>(len); i += stride) {
      dst[i] = fill_val;
    }
  }
}

__device__ __forceinline__ int32_t align_up_block(int32_t count, int32_t block_size) {
  if ((block_size & (block_size - 1)) == 0) {
    return (count + block_size - 1) & ~(block_size - 1);
  }
  return CEILDIV(count, block_size) * block_size;
}

__device__ __forceinline__ void write_expert_block_ids(
    int32_t* __restrict__ expert_ids,
    int32_t start,
    int32_t end,
    int32_t block_size,
    int32_t value) {
  if (start >= end) {
    return;
  }
  int32_t block_begin;
  int32_t block_end;
  if ((block_size & (block_size - 1)) == 0) {
    const int32_t shift = __ffs(block_size) - 1;
    block_begin = start >> shift;
    block_end = end >> shift;
  } else {
    block_begin = start / block_size;
    block_end = end / block_size;
  }
  for (int32_t b = block_begin; b < block_end; ++b) {
    expert_ids[b] = value;
  }
}

/* numel <= smem_size */
template <typename scalar_t, int smem_size, bool Is_EP, int CUB_BLOCK_THREADS>
__global__ void moe_align_block_size_kernel_opt_fused_small_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ expert_map,
    bool* __restrict__ expert_mask, int32_t* __restrict__ num_local_tokens,
    int64_t sorted_token_ids_len, bool Is_fuse_fill, int32_t topk) {
  const int32_t topk_numel = static_cast<int32_t>(numel);
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  if (num_local_tokens != nullptr) {
    const int32_t local_tok = num_local_tokens[0];
    if (local_tok == 0) {
      if (tid == 0) {
        *total_tokens_post_pad = 0;
      }
      return;
    }
    numel = static_cast<size_t>(local_tok) * static_cast<size_t>(topk);
  }
  if (Is_fuse_fill) {
    fill_sorted_token_ids_vec(
        sorted_token_ids, sorted_token_ids_len, topk_numel, tid, stride);
  }

  __shared__ int32_t shared_counts[smem_size + 1];

  // CUB Scan 需要额外的共享内存
  typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ int32_t local_expert_map[smem_size];

  // 初始化
  for (int i = static_cast<int>(tid); i < num_experts; i += static_cast<int>(stride)) {
    shared_counts[i] = 0;
  }
  if (expert_map != nullptr) {
    if (tid < static_cast<size_t>(num_experts)) {
      local_expert_map[tid] = expert_map[tid];
    }
  }
  __syncthreads();

  // 直方图统计
  bool is_valid_expert = true;
  int expert_id = 0;

  if (tid < numel) {
    expert_id = static_cast<int>(topk_ids[tid]);
    if (expert_map != nullptr && local_expert_map[expert_id] == -1) {
      is_valid_expert = false;
    }

    if constexpr (Is_EP) {
      if (is_valid_expert) {
        atomicAdd(&shared_counts[expert_id], 1);
      }
    } else {
      atomicAdd(&shared_counts[expert_id], 1);
    }
  }
  __syncthreads();

  // Padding & 执行 CUB 扫描
  int32_t my_padded_count = 0;
  if (tid < static_cast<size_t>(num_experts)) {
    my_padded_count = align_up_block(shared_counts[tid], block_size);
  }
  int32_t exclusive_prefix_sum = 0;
  cub::Sum scan_op;

  BlockScan(temp_storage).ExclusiveScan(
      my_padded_count,
      exclusive_prefix_sum,
      0,
      scan_op);

  if (tid == CUB_BLOCK_THREADS - 1) {
    const int32_t total = exclusive_prefix_sum + my_padded_count;
    *total_tokens_post_pad = total;
    shared_counts[num_experts] = total;
  }

  if (tid < static_cast<size_t>(num_experts)) {
    shared_counts[tid] = exclusive_prefix_sum;
  }
  __syncthreads();
  if (tid < static_cast<size_t>(num_experts)) {
    const int32_t mapped =
        (expert_map != nullptr) ? local_expert_map[tid] : static_cast<int32_t>(tid);
    write_expert_block_ids(
        expert_ids, shared_counts[tid], shared_counts[tid + 1], block_size, mapped);
  }
  __syncthreads();
  if (tid < numel) {
    if constexpr (Is_EP) {
      if (is_valid_expert) {
        int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
        sorted_token_ids[rank_post_pad] = static_cast<int32_t>(tid);
      }
    } else {
      int32_t rank_post_pad = atomicAdd(&shared_counts[expert_id], 1);
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(tid);
    }
    if (expert_mask != nullptr) {
      expert_mask[tid] = is_valid_expert;
    }
  }
}

template <typename scalar_t, int smem_size, int CUB_BLOCK_THREADS>
__global__ void moe_align_block_size_kernel_opt_fused_cub(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ expert_map,
    bool* __restrict__ expert_mask, int32_t* __restrict__ num_local_tokens,
    bool Is_EP, int64_t sorted_token_ids_len, bool Is_fuse_fill, int32_t topk) {
  static_assert(CUB_BLOCK_THREADS >= smem_size,
                "CUB_BLOCK_THREADS must be >= smem_size");
  const int32_t topk_numel = static_cast<int32_t>(numel);
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  if (num_local_tokens != nullptr) {
    const int32_t local_tok = num_local_tokens[0];
    if (local_tok == 0) {
      if (tid == 0) {
        *total_tokens_post_pad = 0;
      }
      return;
    }
    numel = static_cast<size_t>(local_tok) * static_cast<size_t>(topk);
  }

  if (Is_fuse_fill) {
    fill_sorted_token_ids_vec(
        sorted_token_ids, sorted_token_ids_len, topk_numel, tid, stride);
  }

  constexpr int HIST_COPIES = hcu_align_hist_copies<smem_size>();
  constexpr int HIST_STRIDE = smem_size + 1;
  __shared__ int32_t shared_hist[HIST_COPIES * HIST_STRIDE];
  __shared__ int32_t shared_offsets[smem_size + 1];
  __shared__ int32_t local_expert_map[smem_size];
  typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = static_cast<int>(tid); i < HIST_COPIES * HIST_STRIDE;
       i += static_cast<int>(stride)) {
    shared_hist[i] = 0;
  }
  if (expert_map != nullptr && tid < static_cast<size_t>(num_experts)) {
    local_expert_map[tid] = expert_map[tid];
  }
  __syncthreads();

  const int copy_id =
      (HIST_COPIES > 1) ? (static_cast<int>(tid) & (HIST_COPIES - 1)) : 0;
  int32_t* my_hist = shared_hist + copy_id * HIST_STRIDE;

  for (size_t i = tid; i < numel; i += stride) {
    const int expert_id = static_cast<int>(topk_ids[i]);
    const bool is_valid =
        (expert_map == nullptr) || (local_expert_map[expert_id] != -1);
    if (expert_mask != nullptr) {
      expert_mask[i] = is_valid;
    }
    if (!Is_EP || is_valid) {
      atomicAdd(&my_hist[expert_id], 1);
    }
  }

  __syncthreads();

  int32_t local_prefix[HIST_COPIES];
  int32_t expert_count = 0;
  if (tid < static_cast<size_t>(num_experts)) {
#pragma unroll
    for (int c = 0; c < HIST_COPIES; ++c) {
      local_prefix[c] = expert_count;
      expert_count += shared_hist[c * HIST_STRIDE + tid];
    }
  }

  const int32_t my_padded_count =
      (tid < static_cast<size_t>(num_experts))
          ? align_up_block(expert_count, block_size)
          : 0;
  int32_t exclusive_prefix_sum = 0;
  cub::Sum scan_op;

  BlockScan(temp_storage).ExclusiveScan(
      my_padded_count,
      exclusive_prefix_sum,
      0,
      scan_op);

  if (tid == CUB_BLOCK_THREADS - 1) {
    const int32_t total = exclusive_prefix_sum + my_padded_count;
    *total_tokens_post_pad = total;
    shared_offsets[num_experts] = total;
  }

  if (tid < static_cast<size_t>(num_experts)) {
    shared_offsets[tid] = exclusive_prefix_sum;
#pragma unroll
    for (int c = 0; c < HIST_COPIES; ++c) {
      shared_hist[c * HIST_STRIDE + tid] = exclusive_prefix_sum + local_prefix[c];
    }
  }

  __syncthreads();

  if (tid < static_cast<size_t>(num_experts)) {
    const int32_t mapped =
        (expert_map != nullptr) ? local_expert_map[tid] : static_cast<int32_t>(tid);
    write_expert_block_ids(
        expert_ids, shared_offsets[tid], shared_offsets[tid + 1], block_size, mapped);
  }

  // Pass 2 scatter only accesses shared_hist and sorted_token_ids (disjoint from
  // shared_offsets and expert_ids above).
  for (size_t i = tid; i < numel; i += stride) {
    const int expert_id = static_cast<int>(topk_ids[i]);
    if (!Is_EP || expert_map == nullptr || local_expert_map[expert_id] != -1) {
      const int32_t rank_post_pad = atomicAdd(&my_hist[expert_id], 1);
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
    }
  }
}

template <typename scalar_t, int smem_size, int CUB_BLOCK_THREADS>
__global__ void moe_align_block_size_kernel_opt_cub(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum,
    int32_t* __restrict__ expert_map, bool* __restrict__ expert_mask,
    int32_t* __restrict__ num_local_tokens, bool Is_EP,
    int64_t sorted_token_ids_len, bool Is_fuse_fill,
    int32_t* __restrict__ global_count, int32_t topk) {
  static_assert(CUB_BLOCK_THREADS >= smem_size,
                "CUB_BLOCK_THREADS must be >= smem_size");
  const int32_t topk_numel = static_cast<int32_t>(numel);
  const int tid = threadIdx.x;

  size_t effective_numel = numel;
  if (num_local_tokens != nullptr) {
    const int32_t local_tok = num_local_tokens[0];
    if (local_tok == 0) {
      if (blockIdx.x == 0 && tid == 0) {
        *total_tokens_post_pad = 0;
      }
      return;
    }
    effective_numel = static_cast<size_t>(local_tok) * static_cast<size_t>(topk);
  }

  const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
  const size_t global_stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  constexpr int HIST_COPIES = hcu_align_hist_copies<smem_size>();
  constexpr int HIST_STRIDE = smem_size + 1;
  __shared__ int32_t shared_hist[HIST_COPIES * HIST_STRIDE];
  __shared__ int32_t shared_offsets[smem_size + 1];
  __shared__ int32_t local_expert_map[smem_size];

  for (int i = tid; i < HIST_COPIES * HIST_STRIDE; i += blockDim.x) {
    shared_hist[i] = 0;
  }
  if (expert_map != nullptr && tid < num_experts) {
    local_expert_map[tid] = expert_map[tid];
  }

  if (Is_fuse_fill) {
    fill_sorted_token_ids_vec(
        sorted_token_ids, sorted_token_ids_len, topk_numel, global_tid, global_stride);
  }

  typedef cub::BlockScan<int32_t, CUB_BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __syncthreads();

  const int copy_id = (HIST_COPIES > 1) ? (tid & (HIST_COPIES - 1)) : 0;
  int32_t* my_hist = shared_hist + copy_id * HIST_STRIDE;

  for (size_t i = global_tid; i < effective_numel; i += global_stride) {
    const int expert_id = static_cast<int>(topk_ids[i]);
    const bool is_valid =
        (expert_map == nullptr) || (local_expert_map[expert_id] != -1);
    if (expert_mask != nullptr) {
      expert_mask[i] = is_valid;
    }
    if (!Is_EP || is_valid) {
      atomicAdd(&my_hist[expert_id], 1);
    }
  }

  __syncthreads();
  if (tid < num_experts) {
    int32_t count = 0;
#pragma unroll
    for (int c = 0; c < HIST_COPIES; ++c) {
      count += shared_hist[c * HIST_STRIDE + tid];
    }
    if (count > 0) {
      atomicAdd(&cumsum[tid], count);
    }
  }

  // Device-wide fence + block barrier BEFORE signaling global_count so all waves
  // across all CUs / XCDs (e.g. 4-XCD gfx946) publish cumsum updates first.
  __threadfence();
  __syncthreads();

  __shared__ bool is_last_block;
  if (tid == 0) {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
    const int ticket = __hip_atomic_fetch_add(
        global_count, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#else
    const int ticket = atomicAdd(global_count, 1);
#endif
    is_last_block = (ticket == (static_cast<int>(gridDim.x) - 1));
  }
  __syncthreads();

  if (!is_last_block) {
    return;
  }

  // Acquire device-coherent view across all consumer waves & XCDs.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
  (void)__hip_atomic_load(
      global_count, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#else
  __threadfence();
#endif
  int32_t my_padded_count = 0;
  if (tid < num_experts) {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
    const int32_t expert_count = __hip_atomic_load(
        &cumsum[tid], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    const int32_t expert_count = atomicAdd(&cumsum[tid], 0);
#endif
    my_padded_count = align_up_block(expert_count, block_size);
  }
  int32_t exclusive_prefix_sum = 0;
  cub::Sum scan_op;

  BlockScan(temp_storage).ExclusiveScan(
      my_padded_count,
      exclusive_prefix_sum,
      0,
      scan_op);

  if (tid == CUB_BLOCK_THREADS - 1) {
    const int32_t total = exclusive_prefix_sum + my_padded_count;
    *total_tokens_post_pad = total;
    shared_offsets[num_experts] = total;
  }

  if (tid < num_experts) {
    shared_offsets[tid] = exclusive_prefix_sum;
  }
  __syncthreads();

  if (tid < num_experts) {
    const int32_t mapped =
        (expert_map != nullptr) ? local_expert_map[tid] : tid;
    write_expert_block_ids(
        expert_ids, shared_offsets[tid], shared_offsets[tid + 1], block_size, mapped);
    cumsum[tid] = shared_offsets[tid];
  }
}


template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum,
    int32_t *__restrict__ expert_map) {
  extern __shared__ int32_t shared_counts[];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i]; // Original code for no map
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];

      cumsum[i] =
          cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      if (expert_map) {
        expert_ids[i / block_size] = expert_map[threadIdx.x];
      } else {
        expert_ids[i / block_size] = threadIdx.x;
      }
    }
  }
}

template <typename scalar_t, int TOPK>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ cumsum_buffer,
    size_t numel, int32_t num_experts,
    int32_t *__restrict__ num_local_tokens) {
  if (num_local_tokens != nullptr) {
    if (num_local_tokens[0] == 0) {
      return;
    } else {
      numel = num_local_tokens[0] * TOPK;
    }
  }
  int expert_id;
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  //const size_t stride = blockDim.x * gridDim.x;
  __shared__ int32_t lds_count[256];
  if (threadIdx.x < num_experts) {
    lds_count[threadIdx.x] = 0;
  }
  __syncthreads();
  if(tid < numel){
    expert_id = topk_ids[tid];
    atomicAdd(&lds_count[expert_id], 1);
  }
  __syncthreads();
  if (threadIdx.x < num_experts) {
    lds_count[threadIdx.x] =
        atomicAdd(&cumsum_buffer[threadIdx.x], lds_count[threadIdx.x]);
  }
  // for (size_t i = threadIdx.x; i < 256; i += blockDim.x) {
    
  // }
  __syncthreads();
  if(tid < numel){
    int32_t rank_post_pad = atomicAdd(&lds_count[expert_id], 1);
     sorted_token_ids[rank_post_pad] = tid;
  }
}

template <typename scalar_t, int smem_size>
__global__ void count_and_sort_expert_tokens_kernel_EP(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    size_t numel, int32_t num_experts,
    int32_t* __restrict__ num_local_tokens, int32_t* __restrict__ expert_map,
    bool Is_EP, int32_t topk) {
  if (num_local_tokens != nullptr) {
    const int32_t local_tok = num_local_tokens[0];
    if (local_tok == 0) {
      return;
    }
    numel = static_cast<size_t>(local_tok) * static_cast<size_t>(topk);
  }
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const int ltid = threadIdx.x;

  constexpr int HIST_COPIES = hcu_align_hist_copies<smem_size>();
  constexpr int HIST_STRIDE = smem_size + 1;
  __shared__ int32_t shared_hist[HIST_COPIES * HIST_STRIDE];
  __shared__ int32_t local_expert_map[smem_size];

  for (int i = ltid; i < HIST_COPIES * HIST_STRIDE; i += blockDim.x) {
    shared_hist[i] = 0;
  }
  if (expert_map != nullptr && ltid < num_experts) {
    local_expert_map[ltid] = expert_map[ltid];
  }
  __syncthreads();

  const int copy_id = (HIST_COPIES > 1) ? (ltid & (HIST_COPIES - 1)) : 0;
  int32_t* my_hist = shared_hist + copy_id * HIST_STRIDE;

  for (size_t i = tid; i < numel; i += stride) {
    const int expert_id = static_cast<int>(topk_ids[i]);
    if (!Is_EP || expert_map == nullptr || local_expert_map[expert_id] != -1) {
      atomicAdd(&my_hist[expert_id], 1);
    }
  }
  __syncthreads();

  // cumsum_buffer is indexed by logical expert_id (matching Kernel 1).
  if (ltid < num_experts) {
    int32_t local_prefix[HIST_COPIES];
    int32_t block_total = 0;
#pragma unroll
    for (int c = 0; c < HIST_COPIES; ++c) {
      local_prefix[c] = block_total;
      block_total += shared_hist[c * HIST_STRIDE + ltid];
    }
    if (block_total > 0) {
      const int32_t global_base = atomicAdd(&cumsum_buffer[ltid], block_total);
#pragma unroll
      for (int c = 0; c < HIST_COPIES; ++c) {
        shared_hist[c * HIST_STRIDE + ltid] = global_base + local_prefix[c];
      }
    }
  }

  __syncthreads();
  for (size_t i = tid; i < numel; i += stride) {
    const int expert_id = static_cast<int>(topk_ids[i]);
    if (!Is_EP || expert_map == nullptr || local_expert_map[expert_id] != -1) {
      const int32_t rank_post_pad = atomicAdd(&my_hist[expert_id], 1);
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
    }
  }
}
template <typename scalar_t, int TOPK, bool Is_bf16>
__global__ void
moe_sum_kernel(scalar_t *__restrict__ out,         // [..., d]
               const scalar_t *__restrict__ input, // [..., topk, d]
               const int d, scalar_t *__restrict__ bias, float factor) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    if (!Is_bf16)
      factor = 1.0f / factor;
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += input[token_idx * TOPK * d + k * d + idx];
    }
    if (bias == nullptr) {
      out[token_idx * d + idx] = x * factor;
    } else {
      if (Is_bf16) {
        out[token_idx * d + idx] = x * factor + bias[token_idx * d + idx];
      } else {
        out[token_idx * d + idx] = x + bias[token_idx * d + idx] * factor;
      }
    }
  }
}

template <typename scalar_t>
static __device__ inline void moe_sum_from_float(scalar_t &out, float f) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    out = f;
  } else {
    union {
      float fp32;
      uint32_t int32;
    } u = {f};
    u.int32 += 0x7fff + ((u.int32 >> 16) & 1);
    out = u.int32 >> 16;
  }
}
 
template <typename scalar_t>
static __device__ inline float moe_sum_to_float(scalar_t in) {
  if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
    return in;
  } else {
    union {
      uint32_t int32;
      float fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}

template <int TOPK, int VEC_SIZE, int D_VEC,
          int total_threads_per_token, int TILE_WIDTH>
__global__ void moe_sum_kernel_reg_bf16_no_bias_vec(
    uint16_t *__restrict__ out, const uint16_t *__restrict__ input,
    float factor, int num_tokens) {
  using VecType = Vector<uint16_t, VEC_SIZE>;
  for (int64_t token_idx = blockIdx.y; token_idx < num_tokens;
       token_idx += gridDim.y) {
    const VecType *input_base =
        reinterpret_cast<const VecType *>(input) + token_idx * TOPK * D_VEC;
    VecType *out_base = reinterpret_cast<VecType *>(out) + token_idx * D_VEC;

    const int tid = threadIdx.x;
    for (int vec_idx = blockIdx.x * TILE_WIDTH + tid; vec_idx < D_VEC;
         vec_idx += total_threads_per_token) {
      float acc[VEC_SIZE] = {0.0f};
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        VecType loaded_vec = input_base[k * D_VEC + vec_idx];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += moe_sum_to_float(loaded_vec.val[i]);
        }
      }

      VecType result_vec;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        moe_sum_from_float(result_vec.val[i], acc[i] * factor);
      }
      out_base[vec_idx] = result_vec;
    }
  }
}

template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,
          int total_threads_per_token, int TILE_WIDTH, bool Is_bf16>
__global__ void moe_sum_kernel_reg(scalar_t *__restrict__ out,
                                   const scalar_t *__restrict__ input,
                                   scalar_t *__restrict__ bias, float factor,
                                   bool *expert_mask,int32_t *num_local_tokens) {
  using VecType = Vector<scalar_t, VEC_SIZE>;
  const int token_idx = blockIdx.y;
   if(num_local_tokens != nullptr){
    if(token_idx > num_local_tokens[0])  return;
  }
  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;
  if (expert_mask != nullptr) {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        VecType sum_vec = {0.f};
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (expert_mask[token_idx * TOPK + k]) {
            VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] += current_vec.val[i];
            }
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
    }
  } else {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        VecType sum_vec = input_base_ptr[0 * d_vec + current_idx];
#pragma unroll
        for (int k = 1; k < TOPK; ++k) {
          VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec.val[i] += current_vec.val[i];
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
    }
  }
}
template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,
          int total_threads_per_token, int TILE_WIDTH, bool Is_bf16>
__global__ void moe_sum_kernel_reg_bf16_add_bias(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    scalar_t *__restrict__ bias, float factor, bool *expert_mask,int32_t *num_local_tokens,int num_tokens) {
  int token_idx = blockIdx.y;
  int token_num = num_tokens;   
  if(num_local_tokens != nullptr){
    if(token_idx > num_local_tokens[0])  return;
    token_num = std::min(token_num,num_local_tokens[0]);
  }
  using VecType = Vector<scalar_t, VEC_SIZE>;
  for(token_idx = blockIdx.y;token_idx < token_num; token_idx += gridDim.y){
      const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;
  if (expert_mask != nullptr) {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        Vector<float, VEC_SIZE> sum_vec_f32 = {0.f};
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (expert_mask[token_idx * TOPK + k]) {
            VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec_f32.val[i] += static_cast<float>(current_vec.val[i]);
            }
          }
        }
        VecType sum_vec_rounded;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
          sum_vec_rounded.val[i] = static_cast<scalar_t>(sum_vec_f32.val[i]);
        }
        if (bias != nullptr) {
          VecType temp_mul_rounded;
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(temp_mul_rounded.val[i]) +
                static_cast<float>(bias_val.val[i]));
          }
          out_base_ptr[current_idx] = sum_vec_rounded;
        } else {
          VecType temp_mul_rounded;
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
          out_base_ptr[current_idx] = temp_mul_rounded;
        }
      }
    }
  } else {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {
      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
        Vector<float, VEC_SIZE> sum_vec_f32;
        VecType initial_vec = input_base_ptr[0 * d_vec + current_idx];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
          sum_vec_f32.val[i] = static_cast<float>(initial_vec.val[i]);
        }

#pragma unroll
        for (int k = 1; k < TOPK; ++k) {
          VecType current_vec = input_base_ptr[k * d_vec + current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec_f32.val[i] += static_cast<float>(current_vec.val[i]);
          }
        }
        VecType sum_vec_rounded;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
          sum_vec_rounded.val[i] = static_cast<scalar_t>(sum_vec_f32.val[i]);
        }
        if (bias != nullptr) {
          VecType temp_mul_rounded;
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            sum_vec_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(temp_mul_rounded.val[i]) +
                static_cast<float>(bias_val.val[i]));
          }
          out_base_ptr[current_idx] = sum_vec_rounded;
        } else {
          VecType temp_mul_rounded;
#pragma unroll
          for (int i = 0; i < VEC_SIZE; i++) {
            temp_mul_rounded.val[i] = static_cast<scalar_t>(
                static_cast<float>(sum_vec_rounded.val[i]) * factor);
          }
          out_base_ptr[current_idx] = temp_mul_rounded;
        }
      }
    }
  }
  }
}


template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec, 
          int total_threads_per_token, int TILE_WIDTH, 
          bool HasBias>
__global__ void moe_sum_kernel_reg_bf16_add_bias_opt(
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ bias,
    float factor,
    int num_tokens) {

    int64_t token_idx = blockIdx.y;
    using VecType = Vector<scalar_t, VEC_SIZE>;
    for(;token_idx < num_tokens; token_idx += gridDim.y){
          const int64_t token_offset_input = token_idx * TOPK * d_vec;
          const int64_t token_offset_out   = token_idx * d_vec;
         
          const VecType* input_base = reinterpret_cast<const VecType*>(input + token_offset_input);
          VecType* out_base = reinterpret_cast<VecType*>(out + token_offset_out);
          
          const VecType* bias_base = nullptr;
          if constexpr (HasBias) {
              bias_base = reinterpret_cast<const VecType*>(bias + token_offset_out);
          }

          const int tid = threadIdx.x;
          const int num_vectors = d_vec / VEC_SIZE;
          for (int tile_idx = blockIdx.x * TILE_WIDTH; tile_idx < num_vectors; tile_idx += total_threads_per_token) {
              const int vec_idx = tile_idx + tid;
              
              if (vec_idx < num_vectors) { 
                  float acc[VEC_SIZE] = {0.0f};

                  #pragma unroll VEC_SIZE
                  for (int k = 0; k < TOPK; ++k) {
                    int input_vec_idx = k * num_vectors + vec_idx;
                    VecType loaded_vec = input_base[input_vec_idx];
                    #pragma unroll VEC_SIZE
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        acc[i] += static_cast<float>(loaded_vec.val[i]);
                    }
                  }

                  VecType result_vec;
                  #pragma unroll VEC_SIZE
                  for(int i = 0; i < VEC_SIZE; i++){
                      result_vec.val[i] = static_cast<scalar_t>(acc[i]);
                  }
                  if constexpr (HasBias) {
                      VecType b_val = bias_base[vec_idx];
                      #pragma unroll VEC_SIZE
                      for (int i = 0; i < VEC_SIZE; ++i) {
                          result_vec.val[i]  = static_cast<scalar_t>(static_cast<float>(result_vec.val[i]) * factor);
                          result_vec.val[i] = static_cast<scalar_t>(static_cast<float>(result_vec.val[i]) + static_cast<float>(b_val.val[i]));
                      }
                  } else {
                      #pragma unroll VEC_SIZE
                      for (int i = 0; i < VEC_SIZE; ++i) {
                          result_vec.val[i]  = static_cast<scalar_t>(static_cast<float>(result_vec.val[i]) * factor);
                      }
                  }
                  out_base[vec_idx] = result_vec;
              }
          }
    }
    
}
template <typename scalar_t, int TOPK, int VEC_SIZE, int d_vec,
          int total_threads_per_token, int TILE_WIDTH, bool Is_bf16>
__global__ void moe_sum_kernel_shared_mem(scalar_t *__restrict__ out,
                                          const scalar_t *__restrict__ input,
                                          scalar_t *__restrict__ bias,
                                          float factor, bool *expert_mask,int32_t* num_local_tokens) {
  using VecType = Vector<scalar_t, VEC_SIZE>;
  __shared__ VecType tile[TOPK][TILE_WIDTH];
  const int token_idx = blockIdx.y;
    if(num_local_tokens != nullptr){
    if(token_idx > num_local_tokens[0])  return;
  }
  const VecType *input_base_ptr =
      reinterpret_cast<const VecType *>(input) + token_idx * TOPK * d_vec;
  VecType *out_base_ptr = reinterpret_cast<VecType *>(out) + token_idx * d_vec;

  const int tid_in_block = threadIdx.x;
  if (expert_mask != nullptr) {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {

      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          tile[k][tid_in_block] = input_base_ptr[k * d_vec + current_idx];
        }
      }
      __syncthreads();
      if (current_idx < d_vec) {
        VecType sum_vec = {0.f};
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (expert_mask[token_idx * TOPK + k]) {
            VecType current_vec = tile[k][tid_in_block];
#pragma unroll
            for (int i = 0; i < 8; i++) {
              sum_vec.val[i] += current_vec.val[i];
            }
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
      __syncthreads();
    }
  } else {
    for (int tile_start_idx = blockIdx.x * TILE_WIDTH; tile_start_idx < d_vec;
         tile_start_idx += total_threads_per_token) {

      const int current_idx = tile_start_idx + tid_in_block;
      if (current_idx < d_vec) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          tile[k][tid_in_block] = input_base_ptr[k * d_vec + current_idx];
        }
      }
      __syncthreads();
      if (current_idx < d_vec) {
        VecType sum_vec = tile[0][tid_in_block];
#pragma unroll
        for (int k = 1; k < TOPK; ++k) {
          VecType current_vec = tile[k][tid_in_block];
#pragma unroll
          for (int i = 0; i < 8; i++) {
            sum_vec.val[i] += current_vec.val[i];
          }
        }
        if (bias == nullptr) {
          for (int i = 0; i < VEC_SIZE; i++) {
             sum_vec.val[i] = sum_vec.val[i] * factor;
          }
          out_base_ptr[current_idx] = sum_vec;
        } else {
          VecType *bias_base_ptr =
              reinterpret_cast<VecType *>(bias) + token_idx * d_vec;
          VecType bias_val = bias_base_ptr[current_idx];
          if (Is_bf16) {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] * factor + bias_val.val[i];
            }
          } else {
            for (int i = 0; i < VEC_SIZE; i++) {
              sum_vec.val[i] = sum_vec.val[i] + bias_val.val[i] * factor;
            }
          }
          out_base_ptr[current_idx] = sum_vec;
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, const int32_t *__restrict__ expert_map) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  extern __shared__ int32_t shared_mem[];
  int32_t *cumsum = shared_mem;
  int32_t *tokens_cnts = (int32_t *)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }

  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id =
          expert_map[logical_expert_id]; // Get physical ID
      ++tokens_cnts[(threadIdx.x + 1) * num_experts + mapped_expert_id];
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
    }
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[i * num_experts + threadIdx.x] +=
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) *
              block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  if (expert_map != nullptr) {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t logical_expert_id = topk_ids[i];
      int32_t mapped_expert_id = expert_map[logical_expert_id];
      int32_t rank_post_pad =
          tokens_cnts[threadIdx.x * num_experts + mapped_expert_id] +
          cumsum[mapped_expert_id];
      sorted_token_ids[rank_post_pad] = i;
      ++tokens_cnts[threadIdx.x * num_experts + mapped_expert_id];
    }
  } else {
    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad =
          tokens_cnts[threadIdx.x * num_experts + expert_id] +
          cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = i;
      ++tokens_cnts[threadIdx.x * num_experts + expert_id];
    }
  }
}
template <typename scalar_t, int TEMPLATE_CAPACITY>
void launch_moe_align_opt(
    const torch::Tensor& topk_ids,
    int64_t num_experts,
    int64_t padded_num_experts,
    int experts_per_warp,
    int64_t block_size,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& experts_ids,
    torch::Tensor& num_tokens_post_pad,
    int32_t* expert_map_ptr,
    bool* expert_mask_ptr,
    int32_t* num_local_tokens_ptr,
    bool Is_EP,
    bool Is_fuse_fill,
    int32_t topk,
    int cu_count,
    bool is_gfx946,
    cudaStream_t stream) {
  const int64_t numel = topk_ids.numel();
  const int64_t sorted_len = sorted_token_ids.numel();
  // On gfx946 (192 CUs / 4 XCDs), enter the multi-block path earlier to avoid
  // idling 191 CUs on medium batches while preserving gfx936/938 thresholds.
  const int64_t multi_block_threshold = is_gfx946 ? 4096 : 16384;

  if (numel <= TEMPLATE_CAPACITY) {
    constexpr int threads = TEMPLATE_CAPACITY;
    if (Is_EP) {
      moe_align_block_size_kernel_opt_fused_small_size<
          scalar_t, TEMPLATE_CAPACITY, true, TEMPLATE_CAPACITY>
          <<<1, threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              static_cast<int32_t>(num_experts),
              static_cast<int32_t>(padded_num_experts),
              experts_per_warp,
              static_cast<int32_t>(block_size),
              static_cast<size_t>(numel),
              expert_map_ptr,
              expert_mask_ptr,
              num_local_tokens_ptr,
              sorted_len,
              Is_fuse_fill,
              topk);
    } else {
      moe_align_block_size_kernel_opt_fused_small_size<
          scalar_t, TEMPLATE_CAPACITY, false, TEMPLATE_CAPACITY>
          <<<1, threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              static_cast<int32_t>(num_experts),
              static_cast<int32_t>(padded_num_experts),
              experts_per_warp,
              static_cast<int32_t>(block_size),
              static_cast<size_t>(numel),
              expert_map_ptr,
              expert_mask_ptr,
              num_local_tokens_ptr,
              sorted_len,
              Is_fuse_fill,
              topk);
    }
  } else if (numel < multi_block_threshold) {
    int threads = 1024;
    if (numel <= 256) {
      threads = 128;
    } else if (numel <= 512) {
      threads = 256;
    } else if (numel <= 1024) {
      threads = 512;
    }
    if (threads < TEMPLATE_CAPACITY) {
      threads = TEMPLATE_CAPACITY;
    }

    auto launch_fused = [&](auto block_threads_tag) {
      constexpr int kBlockThreads = decltype(block_threads_tag)::value;
      moe_align_block_size_kernel_opt_fused_cub<
          scalar_t, TEMPLATE_CAPACITY, kBlockThreads>
          <<<1, kBlockThreads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              static_cast<int32_t>(num_experts),
              static_cast<int32_t>(padded_num_experts),
              experts_per_warp,
              static_cast<int32_t>(block_size),
              static_cast<size_t>(numel),
              expert_map_ptr,
              expert_mask_ptr,
              num_local_tokens_ptr,
              Is_EP,
              sorted_len,
              Is_fuse_fill,
              topk);
    };

    if constexpr (TEMPLATE_CAPACITY <= 128) {
      if (threads == 128) {
        launch_fused(std::integral_constant<int, 128>{});
        return;
      }
    }
    if constexpr (TEMPLATE_CAPACITY <= 256) {
      if (threads == 256) {
        launch_fused(std::integral_constant<int, 256>{});
        return;
      }
    }
    if constexpr (TEMPLATE_CAPACITY <= 512) {
      if (threads == 512) {
        launch_fused(std::integral_constant<int, 512>{});
        return;
      }
    }
    launch_fused(std::integral_constant<int, 1024>{});
  } else {
    constexpr int kK1Threads = (TEMPLATE_CAPACITY > 512) ? 1024 : 512;
    auto options_int =
        torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
    torch::Tensor cumsum_buffer =
        torch::empty({TEMPLATE_CAPACITY + 2}, options_int);
    C10_CUDA_CHECK(cudaMemsetAsync(
        cumsum_buffer.data_ptr<int32_t>(),
        0,
        static_cast<size_t>(TEMPLATE_CAPACITY + 2) * sizeof(int32_t),
        stream));

    const int64_t max_k1_blocks = static_cast<int64_t>(cu_count) * 2;
    const int64_t k1_blocks_64 =
        std::min<int64_t>(max_k1_blocks, (numel + kK1Threads - 1) / kK1Threads);
    const int k1_blocks = static_cast<int>(std::max<int64_t>(k1_blocks_64, 1));

    moe_align_block_size_kernel_opt_cub<scalar_t, TEMPLATE_CAPACITY, kK1Threads>
        <<<k1_blocks, kK1Threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            experts_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(),
            static_cast<int32_t>(num_experts),
            static_cast<int32_t>(padded_num_experts),
            experts_per_warp,
            static_cast<int32_t>(block_size),
            static_cast<size_t>(numel),
            cumsum_buffer.data_ptr<int32_t>(),
            expert_map_ptr,
            expert_mask_ptr,
            num_local_tokens_ptr,
            Is_EP,
            sorted_len,
            Is_fuse_fill,
            cumsum_buffer.data_ptr<int32_t>() + TEMPLATE_CAPACITY + 1,
            topk);

    constexpr int kK2Threads = 1024;
    const int64_t raw_k2_blocks = (numel + kK2Threads - 1) / kK2Threads;
    const int64_t max_k2_blocks =
        std::min<int64_t>(65535, static_cast<int64_t>(cu_count) * 4);
    const int k2_blocks = static_cast<int>(
        std::max<int64_t>(std::min<int64_t>(raw_k2_blocks / 2, max_k2_blocks), 1));

    count_and_sort_expert_tokens_kernel_EP<scalar_t, TEMPLATE_CAPACITY>
        <<<k2_blocks, kK2Threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            static_cast<size_t>(numel),
            static_cast<int32_t>(num_experts),
            num_local_tokens_ptr,
            expert_map_ptr,
            Is_EP,
            topk);
  }
}

void moe_align_block_size(
    torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    const torch::optional<torch::Tensor> &expert_map =  torch::nullopt,
    const torch::optional<torch::Tensor> &expert_mask =  torch::nullopt, // bool
    const torch::optional<torch::Tensor> &num_local_tokens =  torch::nullopt,bool Is_EP = false,bool Is_fuse_fill = true) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto* dev_props = at::cuda::getCurrentDeviceProperties();
  const int cu_count =
      (dev_props != nullptr && dev_props->multiProcessorCount > 0)
          ? dev_props->multiProcessorCount
          : 80;
  const bool is_gfx946 =
      dev_props != nullptr && std::strncmp(dev_props->gcnArchName, "gfx946", 6) == 0;
  const int32_t topk =
      (topk_ids.dim() >= 2) ? static_cast<int32_t>(topk_ids.size(-1)) : 8;

  int32_t* expert_map_ptr =
      (expert_map.has_value() && expert_map->defined())
          ? expert_map->data_ptr<int32_t>()
          : nullptr;
  bool* expert_mask_ptr =
      (expert_mask.has_value() && expert_mask->defined())
          ? expert_mask->data_ptr<bool>()
          : nullptr;
  int32_t* num_local_tokens_ptr =
      (num_local_tokens.has_value() && num_local_tokens->defined())
          ? num_local_tokens->data_ptr<int32_t>()
          : nullptr;

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  AT_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        int template_capacity = 0;
        if (num_experts > 0 && num_experts <= 64) {
          template_capacity = 64;
        } else if (num_experts > 64 && num_experts <= 128) {
          template_capacity = 128;
        } else if (num_experts > 128 && num_experts <= 256) {
          template_capacity = 256;
        } else if (num_experts > 256 && num_experts <= 512) {
          template_capacity = 512;
        } else if (num_experts > 512 && num_experts <= 1024) {
          template_capacity = 1024;
        }

        if (template_capacity == 64) {
          launch_moe_align_opt<scalar_t, 64>(
              topk_ids, num_experts, padded_num_experts, experts_per_warp,
              block_size, sorted_token_ids, experts_ids, num_tokens_post_pad,
              expert_map_ptr, expert_mask_ptr, num_local_tokens_ptr, Is_EP,
              Is_fuse_fill, topk, cu_count, is_gfx946, stream);
        } else if (template_capacity == 128) {
          launch_moe_align_opt<scalar_t, 128>(
              topk_ids, num_experts, padded_num_experts, experts_per_warp,
              block_size, sorted_token_ids, experts_ids, num_tokens_post_pad,
              expert_map_ptr, expert_mask_ptr, num_local_tokens_ptr, Is_EP,
              Is_fuse_fill, topk, cu_count, is_gfx946, stream);
        } else if (template_capacity == 256) {
          launch_moe_align_opt<scalar_t, 256>(
              topk_ids, num_experts, padded_num_experts, experts_per_warp,
              block_size, sorted_token_ids, experts_ids, num_tokens_post_pad,
              expert_map_ptr, expert_mask_ptr, num_local_tokens_ptr, Is_EP,
              Is_fuse_fill, topk, cu_count, is_gfx946, stream);
        } else if (template_capacity == 512) {
          launch_moe_align_opt<scalar_t, 512>(
              topk_ids, num_experts, padded_num_experts, experts_per_warp,
              block_size, sorted_token_ids, experts_ids, num_tokens_post_pad,
              expert_map_ptr, expert_mask_ptr, num_local_tokens_ptr, Is_EP,
              Is_fuse_fill, topk, cu_count, is_gfx946, stream);
        } else if (template_capacity == 1024) {
          launch_moe_align_opt<scalar_t, 1024>(
              topk_ids, num_experts, padded_num_experts, experts_per_warp,
              block_size, sorted_token_ids, experts_ids, num_tokens_post_pad,
              expert_map_ptr, expert_mask_ptr, num_local_tokens_ptr, Is_EP,
              Is_fuse_fill, topk, cu_count, is_gfx946, stream);
        } else {
          // baseline fallback for num_experts > 1024
          auto align_kernel = moe_align_block_size_kernel<scalar_t>;
          auto options_int = torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(topk_ids.device());
          torch::Tensor cumsum_buffer =
              torch::zeros({num_experts + 1}, options_int);
          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          align_kernel<<<1, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
              padded_num_experts, experts_per_warp, block_size,
              topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
              expert_map_ptr);

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t, 8>;
          sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel(),
              num_experts,
              num_local_tokens_ptr);
        }
      });
}
void moe_sum(torch::Tensor &input,  // [num_tokens, topk, hidden_size]
             torch::Tensor &output, // [num_tokens, hidden_size]
             const torch::optional<torch::Tensor> &bias = torch::nullopt ,
             const torch::optional<torch::Tensor> &expert_mask = torch::nullopt ,
             const torch::optional<torch::Tensor> &num_local_tokens = torch::nullopt ,
             const float factor = 1.f,const int expect_m = -1) {
  const int hidden_size = input.size(-1);
  int  num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  input = input.is_contiguous() ? input : input.contiguous();

#define DISPATCH_BF16_NO_BIAS_VEC(HIDDEN_SIZE_VAL, X_DIM_VAL)                \
  do {                                                                       \
    constexpr int block_size = 128;                                          \
    constexpr int VEC_SIZE = 8;                                              \
    constexpr int CONST_HIDDEN_SIZE = HIDDEN_SIZE_VAL;                       \
    constexpr int CONST_X_DIM = X_DIM_VAL;                                   \
    constexpr int D_VEC = CONST_HIDDEN_SIZE / VEC_SIZE;                      \
    dim3 Grid(CONST_X_DIM, num_tokens);                                      \
    auto *out_ptr = reinterpret_cast<uint16_t *>(                            \
        output.data_ptr<at::BFloat16>());                                    \
    const auto *input_ptr = reinterpret_cast<const uint16_t *>(              \
        input.data_ptr<at::BFloat16>());                                     \
    if (topk == 8) {                                                         \
      moe_sum_kernel_reg_bf16_no_bias_vec<                                   \
          8, VEC_SIZE, D_VEC, CONST_X_DIM * block_size, block_size>          \
          <<<Grid, block_size, 0, stream>>>(out_ptr, input_ptr, factor,      \
                                            num_tokens);                     \
    } else if (topk == 9) {                                                  \
      moe_sum_kernel_reg_bf16_no_bias_vec<                                   \
          9, VEC_SIZE, D_VEC, CONST_X_DIM * block_size, block_size>          \
          <<<Grid, block_size, 0, stream>>>(out_ptr, input_ptr, factor,      \
                                            num_tokens);                     \
    } else {                                                                 \
      moe_sum_kernel_reg_bf16_no_bias_vec<                                   \
          10, VEC_SIZE, D_VEC, CONST_X_DIM * block_size, block_size>         \
          <<<Grid, block_size, 0, stream>>>(out_ptr, input_ptr, factor,      \
                                            num_tokens);                     \
    }                                                                        \
  } while (0)

  if (input.scalar_type() == at::ScalarType::BFloat16 &&
      !bias.has_value() && !expert_mask.has_value() &&
      !num_local_tokens.has_value() &&
      (topk == 8 || topk == 9 || topk == 10)) {
    switch (hidden_size) {
    case 2048:
      DISPATCH_BF16_NO_BIAS_VEC(2048, 2);
      return;
    case 3072:
      DISPATCH_BF16_NO_BIAS_VEC(3072, 3);
      return;
    case 4096:
      DISPATCH_BF16_NO_BIAS_VEC(4096, 4);
      return;
    case 5120:
      DISPATCH_BF16_NO_BIAS_VEC(5120, 5);
      return;
    case 6144:
      DISPATCH_BF16_NO_BIAS_VEC(6144, 6);
      return;
    case 7168:
      DISPATCH_BF16_NO_BIAS_VEC(7168, 7);
      return;
    case 8192:
      DISPATCH_BF16_NO_BIAS_VEC(8192, 8);
      return;
    default:
      break;
    }
  }

#undef DISPATCH_BF16_NO_BIAS_VEC

  switch (topk) {
  case 2:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          if (input.scalar_type() == at::ScalarType::BFloat16) {
            moe_sum_kernel<scalar_t, 2, true><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          } else {
            moe_sum_kernel<scalar_t, 2, false><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          }
        });
    break;

  case 3:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          if (input.scalar_type() == at::ScalarType::BFloat16) {
            moe_sum_kernel<scalar_t, 3, true><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          } else {
            moe_sum_kernel<scalar_t, 3, false><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          }
        });
    break;

  case 4:
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "moe_sum_kernel", [&] {
          if (input.scalar_type() == at::ScalarType::BFloat16) {
            moe_sum_kernel<scalar_t, 4, true><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          } else {
            moe_sum_kernel<scalar_t, 4, false><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                hidden_size,
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                factor);
          }
        });
    break;
  case 8: {
    if (hidden_size == 7168 || hidden_size == 6144) {
      
      #define DISPATCH_7168_6144(HIDDEN_SIZE_VAL, ...) \
        switch (HIDDEN_SIZE_VAL) { \
          case 7168: { \
            constexpr int CONST_HIDDEN_SIZE = 7168; \
            constexpr int CONST_X_DIM = 7; \
            __VA_ARGS__(); \
            break; \
          } \
          case 6144: { \
            constexpr int CONST_HIDDEN_SIZE = 6144; \
            constexpr int CONST_X_DIM = 6; \
            __VA_ARGS__(); \
            break; \
          } \
          default: \
            TORCH_CHECK(false, "Unsupported hidden_size: ", HIDDEN_SIZE_VAL); \
        }

      int gridy = num_tokens;
      if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
          if(expect_m > 0){
            gridy = expect_m;
          }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
              gridy /= 2; 
            } else if(num_tokens >= 4096)
            { 
               gridy /= 4;
            }
          }
      }
      
      const int block_size = 128;
      const int VEC_SIZE = 8;
      
      if (num_tokens <= 512) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
              DISPATCH_7168_6144(hidden_size, [&] {
                  dim3 Grid(CONST_X_DIM, gridy);
                  if (input.scalar_type() == at::ScalarType::BFloat16) {
                    if(expert_mask.has_value() || num_local_tokens.has_value()){
                      moe_sum_kernel_reg_bf16_add_bias<
                        scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,
                        expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                    }else{
                      if(bias.has_value()){
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }else{
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, false><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }
                    }
                  } else {
                    moe_sum_kernel_reg<scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE,
                                       CONST_X_DIM * block_size, block_size, false>
                        <<<Grid, block_size, 0, stream>>>(
                            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                            1.0 / factor,
                            expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                    : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                  }
              });
            });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_shared_mem", [&] {
              DISPATCH_7168_6144(hidden_size, [&] {
                  dim3 Grid(CONST_X_DIM, gridy);
                  if (input.scalar_type() == at::ScalarType::BFloat16) {
                      if(expert_mask.has_value() || num_local_tokens.has_value()){
                      moe_sum_kernel_reg_bf16_add_bias<
                        scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,
                        expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                    }else{
                      /* deepseek */
                      if(bias.has_value()){
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }else{
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, false><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }
                    }
                  } else {
                    moe_sum_kernel_shared_mem<scalar_t, 8, VEC_SIZE,
                                              CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                                              block_size, false>
                        <<<Grid, block_size, 0, stream>>>(
                            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                            1.0 / factor,
                            expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                    : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                  }
              });
            });
      }
    } else if (hidden_size == 2048 || hidden_size == 3072 || hidden_size == 4096 || hidden_size == 8192){   
      #define DISPATCH_HIDDEN_SIZE(HIDDEN_SIZE_VAL, ...)       \
            switch (HIDDEN_SIZE_VAL) {                         \
              case 2048: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 2048;        \
                constexpr int CONST_X_DIM = 2;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 3072: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 3072;        \
                constexpr int CONST_X_DIM = 3;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 4096: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 4096;        \
                constexpr int CONST_X_DIM = 4;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 8192: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 8192;        \
                constexpr int CONST_X_DIM = 8;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              default:                                         \
                TORCH_CHECK(false, "Unsupported hidden_size: ", HIDDEN_SIZE_VAL); \
            }
    int gridy = num_tokens;
    if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
        if(expect_m > 0){
            gridy = expect_m;
        }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
                gridy /= 2; 
            } else if(num_tokens >= 4096) { 
               gridy /= 4;
            }
        }
    }

    if(num_tokens == 288) gridy = 256;
    const int block_size = 128;
    const int VEC_SIZE = 8;
    if (num_tokens <= 512) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
                
                DISPATCH_HIDDEN_SIZE(hidden_size, [&] {
                    dim3 Grid(CONST_X_DIM, gridy);
                    if (input.scalar_type() == at::ScalarType::BFloat16) {
                        if(expert_mask.has_value() || num_local_tokens.has_value()){
                             moe_sum_kernel_reg_bf16_add_bias<
                                scalar_t, 8, VEC_SIZE, 
                                CONST_HIDDEN_SIZE / VEC_SIZE, 
                                CONST_X_DIM * block_size,    
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,
                                num_tokens);
                        } else {
                            if(bias.has_value()){
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            } else {
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, false><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            }
                        }
                    } else {
                        moe_sum_kernel_reg<scalar_t, 8, VEC_SIZE, 
                                           CONST_HIDDEN_SIZE / VEC_SIZE, 
                                           CONST_X_DIM * block_size,    
                                           block_size, false>
                            <<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                1.0 / factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                    }
                }); 
            });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_shared_mem", [&] {
                 DISPATCH_HIDDEN_SIZE(hidden_size, [&] {
                    dim3 Grid(CONST_X_DIM, gridy);
                    
                    if (input.scalar_type() == at::ScalarType::BFloat16) {
                         if(expert_mask.has_value() || num_local_tokens.has_value()){
                            moe_sum_kernel_reg_bf16_add_bias<
                            scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                            block_size, true><<<Grid, block_size, 0, stream>>>(
                            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                            factor,
                            expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr,
                            num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                        } else {
                            if(bias.has_value()){
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,num_tokens);
                            } else {
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 8, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, false><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,num_tokens);
                            }
                        }
                    } else {
                        moe_sum_kernel_shared_mem<scalar_t, 8, VEC_SIZE,
                                                  CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                                                  block_size, false>
                            <<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                1.0 / factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr,
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                    }
                 });
            });
    }        
    }else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
          "moe_sum_kernel", [&] {
            if (input.scalar_type() == at::ScalarType::BFloat16) {
              moe_sum_kernel<scalar_t, 8, true><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            } else {
              moe_sum_kernel<scalar_t, 8, false><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            }
          });
    }
  } break;
  case 9: {
    if (hidden_size == 7168 || hidden_size == 6144) {
      int gridy = num_tokens;
      if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
          if(expect_m > 0){
            gridy = expect_m;
          }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
              gridy /= 2; 
            } else if(num_tokens >= 4096)
            { 
               gridy /= 4;
            }
          }
      }
      
      const int block_size = 128;
      const int VEC_SIZE = 8;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
              DISPATCH_7168_6144(hidden_size, [&] {
                  dim3 Grid(CONST_X_DIM, gridy);
                  if (input.scalar_type() == at::ScalarType::BFloat16) {
                    if(expert_mask.has_value() || num_local_tokens.has_value()){
                      moe_sum_kernel_reg_bf16_add_bias<
                        scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,
                        expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                    }else{
                      if(bias.has_value()){
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }else{
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, false><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }
                    }
                  } else {
                    moe_sum_kernel_reg<scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE,
                                       CONST_X_DIM * block_size, block_size, false>
                        <<<Grid, block_size, 0, stream>>>(
                            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                            1.0 / factor,
                            expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                    : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                  }
              });
            });
    } else if (hidden_size == 2048 || hidden_size == 4096 || hidden_size == 8192){   
      #define DISPATCH_HIDDEN_SIZE_9(HIDDEN_SIZE_VAL, ...)       \
            switch (HIDDEN_SIZE_VAL) {                          \
              case 2048: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 2048;        \
                constexpr int CONST_X_DIM = 2;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 4096: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 4096;        \
                constexpr int CONST_X_DIM = 4;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 8192: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 8192;        \
                constexpr int CONST_X_DIM = 8;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              default:                                         \
                TORCH_CHECK(false, "Unsupported hidden_size: ", HIDDEN_SIZE_VAL); \
            }
    int gridy = num_tokens;
    if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
        if(expect_m > 0){
            gridy = expect_m;
        }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
                gridy /= 2; 
            } else if(num_tokens >= 4096) { 
               gridy /= 4;
            }
        }
    }
    const int block_size = 128;
    const int VEC_SIZE = 8;
    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
                DISPATCH_HIDDEN_SIZE_9(hidden_size, [&] {
                    dim3 Grid(CONST_X_DIM, gridy);
                    if (input.scalar_type() == at::ScalarType::BFloat16) {
                        if(expert_mask.has_value() || num_local_tokens.has_value()){
                             moe_sum_kernel_reg_bf16_add_bias<
                                scalar_t, 9, VEC_SIZE, 
                                CONST_HIDDEN_SIZE / VEC_SIZE, 
                                CONST_X_DIM * block_size,    
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,
                                num_tokens);
                        } else {
                            if(bias.has_value()){
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            } else {
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 9, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, false><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            }
                        }
                    } else {
                        moe_sum_kernel_reg<scalar_t, 9, VEC_SIZE, 
                                           CONST_HIDDEN_SIZE / VEC_SIZE, 
                                           CONST_X_DIM * block_size,    
                                           block_size, false>
                            <<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                1.0 / factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                    }
                }); 
            });        
    }else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
          "moe_sum_kernel", [&] {
            if (input.scalar_type() == at::ScalarType::BFloat16) {
              moe_sum_kernel<scalar_t, 9, true><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            } else {
              moe_sum_kernel<scalar_t, 9, false><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            }
          });
    }
  } break;
  case 10: {
    if (hidden_size == 7168 || hidden_size == 6144) {
      int gridy = num_tokens;
      if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
          if(expect_m > 0){
            gridy = expect_m;
          }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
              gridy /= 2; 
            } else if(num_tokens >= 4096)
            { 
               gridy /= 4;
            }
          }
      }
      
      const int block_size = 128;
      const int VEC_SIZE = 8;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
              DISPATCH_7168_6144(hidden_size, [&] {
                  dim3 Grid(CONST_X_DIM, gridy);
                  if (input.scalar_type() == at::ScalarType::BFloat16) {
                    if(expert_mask.has_value() || num_local_tokens.has_value()){
                      moe_sum_kernel_reg_bf16_add_bias<
                        scalar_t, 10, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,
                        expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,num_tokens);
                    }else{
                      if(bias.has_value()){
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 10, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, true><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }else{
                        moe_sum_kernel_reg_bf16_add_bias_opt<
                        scalar_t, 10, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                        block_size, false><<<Grid, block_size, 0, stream>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                        factor,num_tokens);
                      }
                    }
                  } else {
                    moe_sum_kernel_reg<scalar_t, 10, VEC_SIZE, CONST_HIDDEN_SIZE / VEC_SIZE,
                                       CONST_X_DIM * block_size, block_size, false>
                        <<<Grid, block_size, 0, stream>>>(
                            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                            1.0 / factor,
                            expert_mask.has_value() ? expert_mask->data_ptr<bool>()
                                                    : nullptr, num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                  }
              });
            });
    } else if (hidden_size == 2048 || hidden_size == 4096 || hidden_size == 8192){   
      #define DISPATCH_HIDDEN_SIZE_10(HIDDEN_SIZE_VAL, ...)       \
            switch (HIDDEN_SIZE_VAL) {                          \
              case 2048: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 2048;        \
                constexpr int CONST_X_DIM = 2;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 4096: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 4096;        \
                constexpr int CONST_X_DIM = 4;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              case 8192: {                                     \
                constexpr int CONST_HIDDEN_SIZE = 8192;        \
                constexpr int CONST_X_DIM = 8;                 \
                __VA_ARGS__();                                 \
                break;                                         \
              }                                                \
              default:                                         \
                TORCH_CHECK(false, "Unsupported hidden_size: ", HIDDEN_SIZE_VAL); \
            }
    int gridy = num_tokens;
    if(num_local_tokens.has_value() && input.scalar_type() == at::ScalarType::BFloat16){
        if(expect_m > 0){
            gridy = expect_m;
        }else{
            if(num_tokens >= 2048 && num_tokens < 4096){
                gridy /= 2; 
            } else if(num_tokens >= 4096) { 
               gridy /= 4;
            }
        }
    }
    const int block_size = 128;
    const int VEC_SIZE = 8;
    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "moe_sum_kernel_reg", [&] {
                DISPATCH_HIDDEN_SIZE_10(hidden_size, [&] {
                    dim3 Grid(CONST_X_DIM, gridy);
                    if (input.scalar_type() == at::ScalarType::BFloat16) {
                        if(expert_mask.has_value() || num_local_tokens.has_value()){
                             moe_sum_kernel_reg_bf16_add_bias<
                                scalar_t, 10, VEC_SIZE, 
                                CONST_HIDDEN_SIZE / VEC_SIZE, 
                                CONST_X_DIM * block_size,    
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr,
                                num_tokens);
                        } else {
                            if(bias.has_value()){
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 10, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, true><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            } else {
                                moe_sum_kernel_reg_bf16_add_bias_opt<
                                scalar_t, 10, VEC_SIZE, CONST_HIDDEN_SIZE, CONST_X_DIM * block_size,
                                block_size, false><<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                factor, num_tokens);
                            }
                        }
                    } else {
                        moe_sum_kernel_reg<scalar_t, 10, VEC_SIZE, 
                                           CONST_HIDDEN_SIZE / VEC_SIZE, 
                                           CONST_X_DIM * block_size,    
                                           block_size, false>
                            <<<Grid, block_size, 0, stream>>>(
                                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                                1.0 / factor,
                                expert_mask.has_value() ? expert_mask->data_ptr<bool>() : nullptr, 
                                num_local_tokens.has_value() ? num_local_tokens->data_ptr<int32_t>() : nullptr);
                    }
                }); 
            });        
    }else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
          "moe_sum_kernel", [&] {
            if (input.scalar_type() == at::ScalarType::BFloat16) {
              moe_sum_kernel<scalar_t, 10, true><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            } else {
              moe_sum_kernel<scalar_t, 10, false><<<grid, block, 0, stream>>>(
                  output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  hidden_size,
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  factor);
            }
          });
    }
  } break;

  default:
    at::sum_out(output, input, 1);
    break;
  }
}
} // namespace native
} // namespace at
