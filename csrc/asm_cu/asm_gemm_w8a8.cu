// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "aiter_hip_common.h"
#include "hcu_mmac.h"

#define DEBUG 0

using half_t = __half;
using bhalf_t = __hip_bfloat16;

enum GRAD_TYPE {
  nograd,
  wgrad,
  xgrad
};

/* module function args */
template <typename T>
struct HipFunctionArgs {
  // Size of Tensor's packed dims, in elements
  uint32_t gemm_count;
  uint32_t internalArgs;
  uint32_t internalArgs1;
  unsigned int numWorkGroups;
  unsigned int size0;
  unsigned int size1;
  unsigned int size2;
  unsigned int size3;
  T *dataD;
  T *dataC;
  const int8_t *dataA;
  const int8_t *dataB;
  unsigned int strideD1;
  unsigned int strideD2;
  unsigned int strideC1;
  unsigned int strideC2;
  unsigned int strideA1;
  unsigned int strideA2;
  unsigned int strideB1;
  unsigned int strideB2;
  float alpha;
  float beta;
  const float *scaleA;
  const float *scaleB;
  uint32_t strideAs1J;
  uint32_t strideAs2K;
  uint32_t strideBs1J;
  uint32_t strideBs2K;
  void *debugBuffer;
};

namespace at {
namespace native {

template <typename OutputType>
void _gemm_w8a8_128x256x128_TN_impl(
  const int8_t *A_data, int8_t *B_data,
  float *As_data, float *Bs_data, // scales
  OutputType *bias_data,
  OutputType *C_data,
  OutputType *D_data,
  const int &size_m, const int &size_n, const int &size_k,
  bool isBfloat16, 
  uint32_t strideAs1J, uint32_t strideAs2K,
  uint32_t strideBs1J, uint32_t strideBs2K,
  GRAD_TYPE gradType = nograd) {
  // printf("strideAs1J: %d,strideAs2K: %d,strideBs1J: %d, strideBs2K: %d\n", strideAs1J, strideAs2K, strideBs1J, strideBs2K);
  size_t strideC1 = size_n;
  size_t strideC2 = 0;
  size_t strideA1 = gradType == xgrad ? size_n : size_k;
  size_t strideA2 = 0;
  size_t strideB1 = size_k;
  size_t strideB2 = 0;

  float beta = (C_data != nullptr && gradType == wgrad) ? 1.0 : 0.0;
  // printf("beta : %f\n", beta);
  /* grid sizes */
  size_t localWorkSize[3] = {768, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  unsigned int totalWorkGroups0 = size_n / 128;
  unsigned int totalWorkGroups1 = size_m / 256;
  if (totalWorkGroups0*128 < size_n) { totalWorkGroups0++; }
  if (totalWorkGroups1*256 < size_m) { totalWorkGroups1++; }
  uint32_t numWorkGroups = totalWorkGroups0 * totalWorkGroups1;
  globalWorkSize[0] = numWorkGroups * globalWorkSize[0];

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.gemm_count = 1;
  hipFunctionArgs.internalArgs = 1;
  hipFunctionArgs.internalArgs1 = 16;
  hipFunctionArgs.numWorkGroups = numWorkGroups;
  hipFunctionArgs.size0 = size_n;
  hipFunctionArgs.size1 = size_m;
  hipFunctionArgs.size2 = 1;
  hipFunctionArgs.size3 = size_k;
  hipFunctionArgs.dataD = D_data;
  hipFunctionArgs.dataC = C_data;
  hipFunctionArgs.dataA = B_data;
  hipFunctionArgs.dataB = A_data;
  hipFunctionArgs.strideD1 = strideC1;
  hipFunctionArgs.strideD2 = strideC2;
  hipFunctionArgs.strideC1 = strideC1;
  hipFunctionArgs.strideC2 = strideC2;
  hipFunctionArgs.strideA1 = strideA1;
  hipFunctionArgs.strideA2 = strideA2;
  hipFunctionArgs.strideB1 = strideB1;
  hipFunctionArgs.strideB2 = strideB2;
  hipFunctionArgs.alpha = 1.0;
  hipFunctionArgs.beta = beta;
  hipFunctionArgs.scaleA = Bs_data;
  hipFunctionArgs.scaleB = As_data;
  hipFunctionArgs.strideAs1J = strideAs1J;
  hipFunctionArgs.strideAs2K = strideAs2K;
  hipFunctionArgs.strideBs1J = strideBs1J;
  hipFunctionArgs.strideBs2K = strideBs2K;
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  int debug_size = 768 * 4;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  // printf("%s:%d top_k:%d topk:%f\n", __FILE__, __LINE__, top_k, hipFunctionArgs.topk);
  #endif
  hipFunctionArgs.debugBuffer = debug_device;
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  string fp16_co_file_tn = "gemm_w8a8_block/GEMM_W8A8_I8_PERBLOCK_ACTIVE_ASM_TN_MT128X256X128_WGM1_FP16.co";
  string bf16_co_file_tn = "gemm_w8a8_block/GEMM_W8A8_I8_PERBLOCK_ACTIVE_ASM_TN_MT128X256X128_WGM1_BF16.co";
  string fp16_co_file_wgrad = "gemm_w8a8_block/GEMM_W8A8_I8_PERBLOCK_WGRAD_ASM_TN_MT128X256X128_WGM1_FP16.co";
  string bf16_co_file_wgrad = "gemm_w8a8_block/GEMM_W8A8_I8_PERBLOCK_WGRAD_ASM_TN_MT128X256X128_WGM1_BF16.co";
  string fp16_co_file_xgrad = "gemm_w8a8_block/GEMM_W8A8_I8_PERBLOCK_XGRAD_ASM_NN_MT128X256X128_WGM1_FP16.co";
  string bf16_co_file_xgrad = "gemm_w8a8_block/GEMM_W8A8_I8_PERBLOCK_XGRAD_ASM_NN_MT128X256X128_WGM1_BF16.co";
	static AiterAsmKernel gemm_w8a8_fp16_tn("GEMM_W8A8_I8_PERBLOCK_ACTIVE_ASM_TN_MT128X256X128_WGM1_FP16", fp16_co_file_tn.c_str());
	static AiterAsmKernel gemm_w8a8_bf16_tn("GEMM_W8A8_I8_PERBLOCK_ACTIVE_ASM_TN_MT128X256X128_WGM1_BF16", bf16_co_file_tn.c_str());
  static AiterAsmKernel gemm_w8a8_fp16_wgrad("GEMM_W8A8_I8_PERBLOCK_WGRAD_ASM_TN_MT128X256X128_WGM1_FP16", fp16_co_file_wgrad.c_str());
  static AiterAsmKernel gemm_w8a8_bf16_wgrad("GEMM_W8A8_I8_PERBLOCK_WGRAD_ASM_TN_MT128X256X128_WGM1_BF16", bf16_co_file_wgrad.c_str());
  static AiterAsmKernel gemm_w8a8_fp16_xgrad("GEMM_W8A8_I8_PERBLOCK_XGRAD_ASM_NN_MT128X256X128_WGM1_FP16", fp16_co_file_xgrad.c_str());
  static AiterAsmKernel gemm_w8a8_bf16_xgrad("GEMM_W8A8_I8_PERBLOCK_XGRAD_ASM_NN_MT128X256X128_WGM1_BF16", bf16_co_file_xgrad.c_str());

	AiterAsmKernel *impl_ptr = nullptr;
  if (gradType == nograd) {
    impl_ptr = isBfloat16 ? &gemm_w8a8_bf16_tn : &gemm_w8a8_fp16_tn;
  }else if (gradType == wgrad) {
    impl_ptr = isBfloat16 ? &gemm_w8a8_bf16_wgrad : &gemm_w8a8_fp16_wgrad;
  }else if (gradType == xgrad) {
    impl_ptr = isBfloat16 ? &gemm_w8a8_bf16_xgrad : &gemm_w8a8_fp16_xgrad;
  }else{
    printf("Type not in {forward, wgrad, xgrad}\n");
    return;
  }
	impl_ptr->launch_kernel_ext({
    &hipFunctionArgs,
    &argsSize,
    globalWorkSize[0],           // gdx
    globalWorkSize[1],           // gdy
    globalWorkSize[2],           // gdz
    localWorkSize[0],            // bdx: 12 wv64
    localWorkSize[1],            // bdy
    localWorkSize[2],            // bdz
    stream
  });
    #if DEBUG
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2]);
  printf("\ndebug:\n");
  hipMemcpy(debug_host, debug_device, debug_size, hipMemcpyDeviceToHost);
  using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  // using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / 2; ++i) {
    if (i % 16 == 0) printf("\n");
    // if (i % 16 == 0) printf("\ni:%d ", i / 16);
    // printf("%3d:%10.4f\t", i, p_debug[i]);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    printf("%3d:%6d\t", i, p_debug[i]);
    // printf("%3d\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    // int index = 2 * i;
    // printf("i:%d %10.4f %6d\t", i, p_debug[index], *reinterpret_cast<uint32_t*>(p_debug + index + 1));
    // printf("i:%d bidz:%d sa:%10.4f sb:%10.4f v:%10.4f\t", i, *reinterpret_cast<int32_t*>(p_debug+index), p_debug[index+1], p_debug[index+2], p_debug[index+3]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + index + 4);
    // printf("i:%d scale:%10.4f %10.4f value:%10.4f %10.4f out:%10.4f %10.4f\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ], static_cast<float>(tmp[0]), static_cast<float>(tmp[2])
    //   );
  }
  printf("\n");
  #endif

  return;
}

torch::Tensor gemm_w8a8_asm(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &As,
  torch::Tensor &Bs,
  c10::optional<torch::Tensor> &C,
  c10::optional<torch::Tensor> &D_,
  c10::optional<torch::Tensor> bias,
  torch::Dtype output_dtype, int M, int N, int K, int gradType = 0) {
  
  torch::Tensor D;
  if (D_.has_value()) {
    D = D_.value();
  } else {
    torch::TensorOptions options = torch::TensorOptions().dtype(output_dtype).device(A.device());
    D = torch::empty({M, N}, options);
  }
	const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  if (output_dtype == torch::kFloat16)
    _gemm_w8a8_128x256x128_TN_impl<half_t>(
      A.data_ptr<int8_t>(),
      B.data_ptr<int8_t>(),
      As.data_ptr<float>(),
      Bs.data_ptr<float>(),
      nullptr,
      C.has_value() ? reinterpret_cast<half_t *>(C.value().data_ptr()) : nullptr,
      reinterpret_cast<half_t *>(D.data_ptr()),
      M, N, K, 0, 
      Bs.stride(-2), 0, As.stride(-2), 0,
      static_cast<GRAD_TYPE>(gradType));
  else if (output_dtype == torch::kBFloat16)
    _gemm_w8a8_128x256x128_TN_impl<bhalf_t>(
      A.data_ptr<int8_t>(),
      B.data_ptr<int8_t>(),
      As.data_ptr<float>(),
      Bs.data_ptr<float>(),
      nullptr,
      C.has_value() ? reinterpret_cast<bhalf_t *>(C.value().data_ptr()) : nullptr,
      reinterpret_cast<bhalf_t *>(D.data_ptr()),
      M, N, K, 1, 
      Bs.stride(-2), 0, As.stride(-2), 0, 
      static_cast<GRAD_TYPE>(gradType));
  else
    TORCH_CHECK(false, "gemm_w8a8 only supports outputType in [Float16, BFloat16]");
  return D;
}

// ============================================================================
// gfx946（少伯）原生 UE8M0 块量化 FP8 GEMM（HIP v_mmac_scale + 未来 .co/.so 汇编热插拔）
//
// 架构与设计总览（方案 A：零多余源码文件 / 零 setup.py 修改）：
//   1. 支持能力：
//      - batch >= 1（通过 grid.z = batch * split_k 映射，零主循环开销）
//      - block_size ∈ {128, 64}（模板参数 SCALE_BLK，K 步长固定 BK = 64）
//      - transpose_flag ∈ {"NN", "NT"}（模板参数 B_TRANSPOSED）
//      - 输出 dtype ∈ {BF16, FP16}，可选 bias（[N] 广播或 [batch, N] 逐 batch）
//   2. 热插拔汇编机制（与 gfx938 hsa/gfx938/ 范式同构）：
//      - 启动前先探测 LIGHTOP_ASM_DIR/gemm_w8a8_fp8_block/<stem>.{co,so,hsaco}，
//        不存在时安全返回 nullptr 并无缝回落本文件内 HIP v_mmac_scale 内核（零 exit/报错）；
//      - 支持 LIGHTOP_GFX946_FP8_BLOCKWISE_USE_ASM=0 强制走 HIP 内核做消融对拍。
//   3. 双核族 + 三档 Tile 启发式（面向 192 CUs 与 gfx938 全量 Dense GEMM shape）：
//      - Family 1（WASP Prefill，4 Producer + 4/8 Consumer，Abarrier + MLS bps + Scale Buffer）：
//          * Tier 0 (A1-Large): 256×256×64, 3-Stage, 12 waves（超大 Prefill）
//          * Tier 1 (A1-Std):   128×256×64, 4-Stage, 12 waves（主力 Prefill，如 N=576/2112/4096）
//          * Tier 2 (A2):       128×128×64, 4-Stage,  8 waves（中 batch / 窄 N，可叠加 Split-K）
//      - Family 2（Lite Decode / 短 K，4 waves 协同，2-Stage LDS，低延迟 + Split-K）：
//          * Tier 3: 64×128×64（M <= 64 或 K <= 256，覆盖 K=128/256 短投影）
//          * Tier 4: 32×128×64（16 < M <= 32）
//          * Tier 5: 16×64×64 （Decode M ∈ [1..16]，MLS 64x16_b8 + 最高 16× Split-K）
//   4. 硬件数据流与 LDS 排布（严格核验自 part2-MLS.pdf P11/P62、part3-DS.pdf P21/P32、ISA v1.8）：
//      - A:[M, K]（AT，K 连续）：MLS 64x32_b8 (t=1) -> 32(M)×64(K) 占 2048B；
//        Consumer 用 ds_read_matrix_trans_format_u8(row=3, col=1, alt=0) 读 16×64，
//        后 16 行偏移 +1024B，一次返回 K=0..31（raw[0:1]）与 K=32..63（raw[2:3]）。
//      - B_NT:[N, K]（BN，K 连续）：与 A 完全同构（+1024B 跨 16 行，ds_read_matrix_trans_format_u8）。
//      - B_NN:[K, N]（BT，N 连续）：MLS 64x32_b8 (t=1) -> 32(K)×64(N) 占 2048B；
//        Consumer 用 ds_read_matrix_format_u8(row=2, col=2, alt=0) 读 32(K)×32(N)，
//        按 part2-MLS.pdf P62 官方规范，N=0..31 偏移 +0B、N=32..63 偏移 **+32B**（非 +1024B！）。
//      - Scale Buffer：1 slot = 256B = 16 rows（ISA v1.8 §2.5.9），每行 16B E8M0。
//        Producer wave 0 经 ds_scale_copy_ds2buf 写入，Consumer 在 ds_read_matrix_* 全部完成后
//        再调用 HCU_SET_SCALE_M0（避免 ds_read_matrix_* 覆写 M0 的别名冲突！）。
//      - Epilogue：v_mmac_scale 使用 lit=true, lts=false，每 lane 持有行 row=lane&15 的
//        连续 4 列 col=(lane>>4)*4+[0..3]，直接向量打包 BF16/FP16 合并写回，零 LDS 转置。
// ============================================================================
namespace fp8_blockwise_946 {

// 预留给后期 hsa/gfx946/gemm_w8a8_fp8_block/*.{co,so,hsaco} 汇编核的标准 ABI 参数包
struct Fp8BlockwiseUe8m0AsmArgs {
  void* data_d;
  const uint8_t* data_a;
  const uint8_t* data_b;
  const float* scale_a;
  const float* scale_b;
  const void* bias;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t batch;
  int32_t split_k;
  int32_t block_size;
  int32_t b_transposed;
  int32_t has_bias;
  int32_t bias_batched;
  int32_t _pad;
};

// 安全热插拔汇编探测与缓存：
// - 与原 AiterAsmKernel 静态构造（缺失 .co 即 exit(0)）不同，此处先以 std::filesystem::exists
//   判断文件存在性，并缓存命中/未命中表，缺失时安静返回 nullptr 回落 HIP 内核。
// - 支持 LIGHTOP_GFX946_FP8_BLOCKWISE_USE_ASM=0 显式关闭汇编用于精度/性能消融对比。
static AiterAsmKernel* try_get_asm_kernel(const std::string& kernel_name,
                                          const std::string& rel_stem) {
  const char* use_asm_env = std::getenv("LIGHTOP_GFX946_FP8_BLOCKWISE_USE_ASM");
  if (use_asm_env != nullptr && std::strcmp(use_asm_env, "0") == 0) {
    return nullptr;
  }
  const char* asm_dir_env = std::getenv("LIGHTOP_ASM_DIR");
  if (asm_dir_env == nullptr || asm_dir_env[0] == '\0') {
    return nullptr;
  }

  static std::mutex cache_mu;
  static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> cache;
  static std::unordered_map<std::string, bool> missing;

  std::lock_guard<std::mutex> lock(cache_mu);
  auto it = cache.find(rel_stem);
  if (it != cache.end()) {
    return it->second.get();
  }
  if (missing.find(rel_stem) != missing.end()) {
    return nullptr;
  }

  const std::string base_dir(asm_dir_env);
  static const char* kExts[] = {".co", ".so", ".hsaco"};
  for (const char* ext : kExts) {
    const std::string rel_file = "gemm_w8a8_fp8_block/" + rel_stem + ext;
    const std::string full_path = base_dir + rel_file;
    if (std::filesystem::exists(full_path)) {
      auto ptr = std::make_unique<AiterAsmKernel>(kernel_name.c_str(), rel_file.c_str());
      AiterAsmKernel* raw = ptr.get();
      cache.emplace(rel_stem, std::move(ptr));
      return raw;
    }
  }
  missing.emplace(rel_stem, true);
  return nullptr;
}

#if defined(__gfx946__)

using v2i = int __attribute__((ext_vector_type(2)));
using v4i = int __attribute__((ext_vector_type(4)));
using v4f = float __attribute__((ext_vector_type(4)));
using v2f = float __attribute__((ext_vector_type(2)));
using v2s = short __attribute__((ext_vector_type(2)));
using lds_i8_ptr = char __attribute__((address_space(3))) *;
using lds_i32_ptr = int __attribute__((address_space(3))) *;

// 设备端极值辅助（避免在 namespace at::native 内被 ATen host 版 min/max 遮蔽）
__device__ __forceinline__ int64_t dmin64(int64_t a, int64_t b) { return a < b ? a : b; }
__device__ __forceinline__ int64_t dmax64(int64_t a, int64_t b) { return a > b ? a : b; }
__device__ __forceinline__ int dmin32(int a, int b) { return a < b ? a : b; }

// 将 __shared__ 通用指针转换为显式 LDS address_space(3) 指针（遵循 LightOp C 风格转换惯例）
__device__ __forceinline__ lds_i8_ptr to_lds_i8(uint8_t* p) {
  return (__attribute__((address_space(3))) char*)p;
}
__device__ __forceinline__ lds_i32_ptr to_lds_i32(uint8_t* p) {
  return (__attribute__((address_space(3))) int*)p;
}

// 零浮点提取 UE8M0 8-bit 指数：
// - 正常 2 的幂次 FP32 2^(e-127) (e ∈ [1..254]) 的位模式为 (e << 23)，右移 23 即得 e；
// - 次正规 2^-127（0x00400000）右移 23 自动得 0（对应硬件 E8M0=0）。
__device__ __forceinline__ uint8_t fp32_pow2_to_e8m0(float s) {
  return static_cast<uint8_t>(__float_as_uint(s) >> 23);
}

// 构造 gfx946 MLS 4-SGPR 矩阵描述符 M#（part2-MLS.pdf P10-11）：
// - srsrc[0..1]：48-bit 全局基地址（16B 对齐）
// - srsrc[2]   ：行步长（元素数，b8 下等于字节数，16B 对齐）
// - srsrc[3]   ：[7:0]=m_filter（主序末尾补零元素数），[15:8]=nm_filter（非主序末尾补零行数），
//                gfx946 已移除 mfmt/interleave 位，且 64x32_b8 filter <= 64 无需 m_filter_hi
// - Hazard（ISA v1.8 §4.2.5）：VALU(readfirstlane) 写 SGPR 后立即被 MLS 读取需 5 个等待周期，
//   故插入 `s_nop 4`（5 cycles）。
__device__ __forceinline__ v4i make_mls_srsrc_uniform(
    const uint8_t* ptr, int stride_bytes, int m_filter, int nm_filter) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr) & 0x0000FFFFFFFFFFFFULL;
  int lo = __builtin_amdgcn_readfirstlane(static_cast<int>(addr & 0xFFFFFFFFULL));
  int hi = __builtin_amdgcn_readfirstlane(static_cast<int>(addr >> 32));
  int st = __builtin_amdgcn_readfirstlane(stride_bytes);
  int ctrl = __builtin_amdgcn_readfirstlane(((nm_filter & 0xFF) << 8) | (m_filter & 0xFF));
  asm volatile("s_nop 4" ::: "memory");
  return v4i{lo, hi, st, ctrl};
}

template <typename OutT>
__device__ __forceinline__ float load_bias_elem(const OutT* ptr) {
  if constexpr (std::is_same_v<OutT, bhalf_t>) {
    return __bfloat162float(*ptr);
  } else {
    return __half2float(*ptr);
  }
}

template <typename OutT>
__device__ __forceinline__ uint2 pack_4f_to_out4(v4f c) {
  uint2 res;
  if constexpr (std::is_same_v<OutT, bhalf_t>) {
    auto lo = __builtin_hcu_cvt_pk_bf16_f32(0, c[0], 0, c[1], false);
    auto hi = __builtin_hcu_cvt_pk_bf16_f32(0, c[2], 0, c[3], false);
    res.x = __builtin_bit_cast(uint32_t, lo);
    res.y = __builtin_bit_cast(uint32_t, hi);
  } else {
    __half2 lo = __float22half2_rn(make_float2(c[0], c[1]));
    __half2 hi = __float22half2_rn(make_float2(c[2], c[3]));
    res.x = __builtin_bit_cast(uint32_t, lo);
    res.y = __builtin_bit_cast(uint32_t, hi);
  }
  return res;
}

template <typename OutT>
__device__ __forceinline__ void store_tile_epilogue(
    v4f acc,
    OutT* __restrict__ out_base,
    float* __restrict__ workspace_base,
    const OutT* __restrict__ bias_row,
    int64_t global_m,
    int64_t global_n,
    int64_t M,
    int64_t N,
    int split_k) {
  if (global_m >= M || global_n >= N) {
    return;
  }
  if (split_k > 1) {
    if (global_n + 4 <= N) {
      *reinterpret_cast<float4*>(workspace_base + global_m * N + global_n) =
          make_float4(acc[0], acc[1], acc[2], acc[3]);
    } else {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (global_n + j < N) {
          workspace_base[global_m * N + global_n + j] = acc[j];
        }
      }
    }
    return;
  }

  if (bias_row != nullptr) {
    if (global_n + 4 <= N) {
      v2f b01 = {load_bias_elem(bias_row + global_n + 0),
                 load_bias_elem(bias_row + global_n + 1)};
      v2f b23 = {load_bias_elem(bias_row + global_n + 2),
                 load_bias_elem(bias_row + global_n + 3)};
      v2f c01 = {acc[0], acc[1]};
      v2f c23 = {acc[2], acc[3]};
      c01 = c01 + b01;
      c23 = c23 + b23;
      acc[0] = c01[0];
      acc[1] = c01[1];
      acc[2] = c23[0];
      acc[3] = c23[1];
    } else {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (global_n + j < N) {
          acc[j] += load_bias_elem(bias_row + global_n + j);
        }
      }
    }
  }

  uint2 packed = pack_4f_to_out4<OutT>(acc);
  if (global_n + 4 <= N) {
    *reinterpret_cast<uint2*>(out_base + global_m * N + global_n) = packed;
  } else {
    uint16_t elems[4];
    *reinterpret_cast<uint2*>(elems) = packed;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (global_n + j < N) {
        *reinterpret_cast<uint16_t*>(out_base + global_m * N + global_n + j) = elems[j];
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Family 1: WASP Prefill 内核（4 Producer + 4/8 Consumer，Abarrier + MLS bps + Scale Buffer）
// - Scale Buffer 精确分配：1 slot = 256B = 16 rows（ISA v1.8 §2.5.9），
//   最坏配置（128x256, 4-stage, SCALE_BLK=64）占用 4 × 48 rows = 192 rows = 12 slots。
// - Abarrier 槽位分配：[0..STAGES-1] 为 FULL，[STAGES..2*STAGES-1] 为 EMPTY，
//   全部位于硬件允许的 0..15 范围（STAGES <= 4 -> 最多使用 8 个槽位），退出前经
//   s_ebarrier_sync + s_abarrier_inv 释放。
// ----------------------------------------------------------------------------
template <
    typename OutT,
    int BM,
    int BN,
    int BK,
    int STAGES,
    int SCALE_BLK,
    bool B_TRANSPOSED,
    int PRODUCER_WAVES,
    int CONSUMER_WAVES>
__global__ __launch_bounds__((PRODUCER_WAVES + CONSUMER_WAVES) * 64, 1)
__attribute__((hcu_scale_buffer_size(12)))
void fp8_blockwise_ue8m0_wasp_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    const OutT* __restrict__ bias,
    OutT* __restrict__ D,
    float* __restrict__ workspace,
    int64_t M,
    int64_t N,
    int64_t K,
    int split_k,
    bool bias_batched) {
  static_assert(BK == 64, "BK must be 64");
  static_assert(SCALE_BLK == 128 || SCALE_BLK == 64, "SCALE_BLK must be 128 or 64");
  static_assert(BM == 128 || BM == 256, "WASP BM must be 128 or 256");
  static_assert(BN == 128 || BN == 256, "WASP BN must be 128 or 256");

  constexpr int TOTAL_WAVES = PRODUCER_WAVES + CONSUMER_WAVES;
  constexpr int M_WAVES = 2;
  constexpr int N_WAVES = CONSUMER_WAVES / M_WAVES;
  constexpr int WM = BM / M_WAVES;           // 64 or 128
  constexpr int WN = BN / N_WAVES;           // 64（保证 WN <= SCALE_BLK，单 wave 仅需 1 份 Scale B）
  constexpr int WM_TILES = WM / 16;          // 4 or 8
  constexpr int WN_TILES = WN / 16;          // 4
  constexpr int K_STEPS_PER_SBLK = SCALE_BLK / BK;  // 128->2 拍刷新一次；64->每拍刷新
  // 无竞争 Scale Buffer 环形槽数：必须覆盖 STAGES 级异步在途流水，防止 Producer 覆写 Consumer 在用 scale
  constexpr int NUM_SBUF_SLOTS = (STAGES + K_STEPS_PER_SBLK - 1) / K_STEPS_PER_SBLK;
  constexpr int NUM_B_BLKS = BN / SCALE_BLK;        // 1, 2, or 4
  constexpr int A_STAGE_BYTES = BM * BK;
  constexpr int B_STAGE_BYTES = BN * BK;
  constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
  constexpr int SCALE_LDS_STRIDE = 512;             // >= BM(256) + NUM_B_BLKS(4)*16 = 320B，256B 对齐
  constexpr int TOTAL_LDS_BYTES = STAGES * STAGE_BYTES + NUM_SBUF_SLOTS * SCALE_LDS_STRIDE;

  // 每个 M-wave 独占 8 行（128B 对齐，覆盖最多 WM_TILES=8 个 16-row tile）；
  // 每个 N scale block 独占 8 行基址，Consumer 通过 M0 定位自身 N-block，scale_b_offset 恒为 0。
  constexpr int A_SBUF_ROWS = M_WAVES * 8;          // 16 rows
  constexpr int B_SBUF_ROWS = NUM_B_BLKS * 8;       // 8, 16, or 32 rows
  constexpr int STAGE_SBUF_ROWS = A_SBUF_ROWS + B_SBUF_ROWS;  // 24, 32, or 48 rows
  static_assert(NUM_SBUF_SLOTS * STAGE_SBUF_ROWS <= 12 * 16,
                "WASP Scale Buffer usage exceeds 12 slots (192 rows)");
  static_assert(TOTAL_LDS_BYTES <= 131072, "WASP LDS exceeds gfx946 128 KB limit");

  __shared__ __align__(256) uint8_t smem[TOTAL_LDS_BYTES];

  const int64_t batch_idx = static_cast<int64_t>(blockIdx.z) / split_k;
  const int split_idx = static_cast<int>(blockIdx.z) % split_k;
  const int64_t num_k_blks = (K + SCALE_BLK - 1) / SCALE_BLK;
  const int64_t num_n_blks = (N + SCALE_BLK - 1) / SCALE_BLK;
  const int64_t k_blks_per_split = (num_k_blks + split_k - 1) / split_k;
  const int64_t k_start = static_cast<int64_t>(split_idx) * k_blks_per_split * SCALE_BLK;
  const int64_t k_end = dmin64(K, static_cast<int64_t>(split_idx + 1) * k_blks_per_split * SCALE_BLK);
  if (k_start >= k_end) {
    return;
  }

  // WGM=8 swizzle across (blockIdx.x, blockIdx.y) for L2 reuse.
  const int grid_m = static_cast<int>(gridDim.y);
  const int grid_n = static_cast<int>(gridDim.x);
  constexpr int WGM = 8;
  const int linear_cta = static_cast<int>(blockIdx.y) * grid_n + static_cast<int>(blockIdx.x);
  const int group_size = WGM * grid_n;
  const int group_id = linear_cta / group_size;
  const int first_m = group_id * WGM;
  const int group_m = dmin32(grid_m - first_m, WGM);
  const int in_group = linear_cta % group_size;
  const int cta_m = first_m + (in_group % group_m);
  const int cta_n = in_group / group_m;

  const int64_t m0 = static_cast<int64_t>(cta_m) * BM;
  const int64_t n0 = static_cast<int64_t>(cta_n) * BN;

  const uint8_t* A_b = A + batch_idx * M * K;
  const uint8_t* B_b = B + batch_idx * K * N;
  const float* sa_b = scale_a + batch_idx * M * num_k_blks;
  const float* sb_b = scale_b + batch_idx * num_k_blks * num_n_blks;
  OutT* D_b = D + batch_idx * M * N;
  float* ws_b = (split_k > 1) ? (workspace + (batch_idx * split_k + split_idx) * M * N) : nullptr;
  const OutT* bias_row = (bias == nullptr) ? nullptr : (bias_batched ? (bias + batch_idx * N) : bias);

  const int wave_id = __builtin_amdgcn_readfirstlane(__builtin_hcu_get_wave_id());
  const int lane = threadIdx.x & 63;

  // Initialize 2 * STAGES abarriers: [0..STAGES-1] = FULL, [STAGES..2*STAGES-1] = EMPTY.
  if (wave_id == 0) {
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      __builtin_hcu_s_abarrier_init(s, TOTAL_WAVES);
      __builtin_hcu_s_abarrier_init(STAGES + s, TOTAL_WAVES);
    }
  }
  __builtin_hcu_s_ebarrier_sync(0);

  const int num_k_steps = static_cast<int>((k_end - k_start + BK - 1) / BK);

  if (wave_id < PRODUCER_WAVES) {
    for (int k_step = 0; k_step < num_k_steps; ++k_step) {
      const int stage = k_step % STAGES;
      const int64_t k_cur = k_start + static_cast<int64_t>(k_step) * BK;
      const int valid_k_64 = static_cast<int>(dmin64(BK, k_end - k_cur));
      const int k_filter_64 = BK - valid_k_64;

      if (k_step >= STAGES) {
        const int empty_phase = ((k_step / STAGES) - 1) & 1;
        __builtin_hcu_s_abarrier_arrive(STAGES + stage);
        while (!__builtin_hcu_s_abarrier_try_wait(STAGES + stage, empty_phase)) {}
      }

      uint8_t* a_lds_stage = smem + stage * STAGE_BYTES;
      uint8_t* b_lds_stage = a_lds_stage + A_STAGE_BYTES;

      __builtin_hcu_s_abarrier_seq(stage);

      // Load A[BM, 64] as (BM/32) blocks of 32x64 (2048 B each), bps=true.
      constexpr int A_BLOCKS = BM / 32;
      #pragma unroll
      for (int ab = wave_id; ab < A_BLOCKS; ab += PRODUCER_WAVES) {
        const int64_t row_a = m0 + ab * 32;
        const int valid_m = static_cast<int>(dmax64(0, dmin64(32, M - row_a)));
        if (valid_m > 0) {
          v4i srsrc_a = make_mls_srsrc_uniform(
              A_b + row_a * K + k_cur, static_cast<int>(K), k_filter_64, 32 - valid_m);
          lds_i8_ptr dst_a = to_lds_i8(a_lds_stage + ab * 2048);
          __builtin_hcu_matrix_load_64x32_b8(srsrc_a, dst_a, 0, true, false, false, false, true);
        }
      }

      // Load B tile (2048 B blocks), bps=true.
      if constexpr (B_TRANSPOSED) {
        // NT: B is [N, K] row-major (K contiguous), (BN/32) blocks of 32(N)x64(K).
        constexpr int B_BLOCKS = BN / 32;
        #pragma unroll
        for (int bb = wave_id; bb < B_BLOCKS; bb += PRODUCER_WAVES) {
          const int64_t row_b = n0 + bb * 32;
          const int valid_n = static_cast<int>(dmax64(0, dmin64(32, N - row_b)));
          if (valid_n > 0) {
            v4i srsrc_b = make_mls_srsrc_uniform(
                B_b + row_b * K + k_cur, static_cast<int>(K), k_filter_64, 32 - valid_n);
            lds_i8_ptr dst_b = to_lds_i8(b_lds_stage + bb * 2048);
            __builtin_hcu_matrix_load_64x32_b8(srsrc_b, dst_b, 0, true, false, false, false, true);
          }
        }
      } else {
        // NN: B is [K, N] row-major (N contiguous), (BN/64)*2 blocks of 32(K)x64(N).
        constexpr int B_N64 = BN / 64;
        constexpr int B_BLOCKS = B_N64 * 2;
        #pragma unroll
        for (int bb = wave_id; bb < B_BLOCKS; bb += PRODUCER_WAVES) {
          const int n64_idx = bb / 2;
          const int k32_idx = bb & 1;
          const int64_t col_b = n0 + n64_idx * 64;
          const int64_t row_k = k_cur + k32_idx * 32;
          const int valid_n = static_cast<int>(dmax64(0, dmin64(64, N - col_b)));
          const int valid_k = static_cast<int>(dmax64(0, dmin64(32, k_end - row_k)));
          if (valid_n > 0 && valid_k > 0) {
            v4i srsrc_b = make_mls_srsrc_uniform(
                B_b + row_k * N + col_b, static_cast<int>(N), 64 - valid_n, 32 - valid_k);
            lds_i8_ptr dst_b = to_lds_i8(b_lds_stage + bb * 2048);
            __builtin_hcu_matrix_load_64x32_b8(srsrc_b, dst_b, 0, true, false, false, false, true);
          }
        }
      }

      // Refresh Scale Buffer on SCALE_BLK boundaries (wave 0 only).
      if (wave_id == 0 && (k_step % K_STEPS_PER_SBLK) == 0) {
        const int64_t k_blk = dmin64(k_cur / SCALE_BLK, num_k_blks - 1);
        const int sb_slot = static_cast<int>(k_blk % NUM_SBUF_SLOTS);
        uint8_t* scale_lds = smem + STAGES * STAGE_BYTES + sb_slot * SCALE_LDS_STRIDE;

        // Scale A: BM E8M0 bytes -> scale_lds[0..BM-1]
        #pragma unroll
        for (int r = lane; r < BM; r += 64) {
          const int64_t gm = dmin64(m0 + r, M - 1);
          scale_lds[r] = fp32_pow2_to_e8m0(sa_b[gm * num_k_blks + k_blk]);
        }
        // Scale B: NUM_B_BLKS distinct blocks, broadcast across 16 lanes each.
        if (lane < 16) {
          #pragma unroll
          for (int bi = 0; bi < NUM_B_BLKS; ++bi) {
            const int64_t n_blk = dmin64((n0 + bi * SCALE_BLK) / SCALE_BLK, num_n_blks - 1);
            const float sb_val = B_TRANSPOSED
                                     ? sb_b[n_blk * num_k_blks + k_blk]
                                     : sb_b[k_blk * num_n_blks + n_blk];
            scale_lds[BM + bi * 16 + lane] = fp32_pow2_to_e8m0(sb_val);
          }
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        if (lane < 16) {
          const int base_row = sb_slot * STAGE_SBUF_ROWS;
          #pragma unroll
          for (int mw = 0; mw < M_WAVES; ++mw) {
            #pragma unroll
            for (int r16 = 0; r16 < WM_TILES; ++r16) {
              lds_i32_ptr src_a = to_lds_i32(
                  scale_lds + mw * WM + r16 * 16 + lane);
              __builtin_hcu_ds_scale_copy_ds2buf(src_a, base_row + mw * 8 + r16, 0, 0, 0);
            }
          }
          #pragma unroll
          for (int bi = 0; bi < NUM_B_BLKS; ++bi) {
            lds_i32_ptr src_b = to_lds_i32(
                scale_lds + BM + bi * 16 + lane);
            __builtin_hcu_ds_scale_copy_ds2buf(
                src_b, base_row + A_SBUF_ROWS + bi * 8, 0, 0, 0);
          }
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      }

      __builtin_hcu_s_abarrier_arrive(stage);
    }
  } else {
    const int cwid = wave_id - PRODUCER_WAVES;
    const int m_wave_id = cwid / N_WAVES;
    const int n_wave_id = cwid % N_WAVES;
    const int wm_base = m_wave_id * WM;
    const int wn_base = n_wave_id * WN;
    const int my_b_blk_local = wn_base / SCALE_BLK;

    v4f acc[WM_TILES][WN_TILES];
    #pragma unroll
    for (int mi = 0; mi < WM_TILES; ++mi) {
      #pragma unroll
      for (int ni = 0; ni < WN_TILES; ++ni) {
        acc[mi][ni] = v4f{0.f, 0.f, 0.f, 0.f};
      }
    }

    for (int k_step = 0; k_step < num_k_steps; ++k_step) {
      const int stage = k_step % STAGES;
      const int full_phase = (k_step / STAGES) & 1;
      const int64_t k_cur = k_start + static_cast<int64_t>(k_step) * BK;

      __builtin_hcu_s_abarrier_arrive(stage);
      while (!__builtin_hcu_s_abarrier_try_wait(stage, full_phase)) {}

      uint8_t* a_lds_stage = smem + stage * STAGE_BYTES;
      uint8_t* b_lds_stage = a_lds_stage + A_STAGE_BYTES;

      // 当 WM_TILES == 8 (256x256 档位) 时将 A 分两半加载 (M_CHUNK = 4)，与 B (16 VGPR) +
      // acc (128 VGPR) 错峰复用 16 个 A-frag VGPR，确保 3 waves/SIMD (<=168 VGPR) 下零 scratch spill。
      constexpr int M_CHUNK = (WM_TILES > 4) ? 4 : WM_TILES;
      constexpr int M_CHUNKS = WM_TILES / M_CHUNK;
      v2i b_frag[WN_TILES][2];

      // 1. 先加载当前 stage 的全部 B fragments（4/2 块共 16 VGPR），在整个 K-step 期间跨 M_CHUNKS 复用。
      if constexpr (B_TRANSPOSED) {
        // NT (BN layout): K contiguous -> ds_read_matrix_trans_format_u8
        #pragma unroll
        for (int ni = 0; ni < WN_TILES; ++ni) {
          const int b_row = wn_base + ni * 16;
          const int b_off = (b_row / 32) * 2048 + ((b_row % 32) / 16) * 1024;
          v4i raw = __builtin_hcu_ds_read_matrix_trans_format_u8(
              to_lds_i32(b_lds_stage + b_off), 0, 3, 1, 0);
          b_frag[ni][0] = v2i{raw[0], raw[1]};
          b_frag[ni][1] = v2i{raw[2], raw[3]};
        }
      } else {
        // NN（BT 布局，N 连续）：使用 ds_read_matrix_format_u8 (rearrange_32x32_b8)。
        // 核验自 part2-MLS.pdf P62：matrix_load_64x32_b8 t 搬入 32(K)×64(N) 后，
        // N=0..31 偏移 +0B，N=32..63 偏移 +32B（非 +1024B！）。
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          #pragma unroll
          for (int ni2 = 0; ni2 < WN_TILES / 2; ++ni2) {
            const int b_col = wn_base + ni2 * 32;
            const int b_off = ((b_col / 64) * 2 + ks) * 2048 + ((b_col % 64) / 32) * 32;
            v4i raw = __builtin_hcu_ds_read_matrix_format_u8(
                to_lds_i32(b_lds_stage + b_off), 0, 2, 2, 0);
            b_frag[2 * ni2 + 0][ks] = v2i{raw[0], raw[1]};
            b_frag[2 * ni2 + 1][ks] = v2i{raw[2], raw[3]};
          }
        }
      }

      const int sb_slot = static_cast<int>((k_cur / SCALE_BLK) % NUM_SBUF_SLOTS);
      const int my_row_a = sb_slot * STAGE_SBUF_ROWS + m_wave_id * 8;
      const int my_row_b = sb_slot * STAGE_SBUF_ROWS + A_SBUF_ROWS + my_b_blk_local * 8;

      #pragma unroll
      for (int mc = 0; mc < M_CHUNKS; ++mc) {
        v2i a_frag[M_CHUNK][2];
        // 2. 分块读取当前 M_CHUNK 的 A fragments（AT 布局 -> ds_read_matrix_trans_format_u8）。
        #pragma unroll
        for (int mi_loc = 0; mi_loc < M_CHUNK; ++mi_loc) {
          const int mi = mc * M_CHUNK + mi_loc;
          const int a_row = wm_base + mi * 16;
          const int a_off = (a_row / 32) * 2048 + ((a_row % 32) / 16) * 1024;
          v4i raw = __builtin_hcu_ds_read_matrix_trans_format_u8(
              to_lds_i32(a_lds_stage + a_off), 0, 3, 1, 0);
          a_frag[mi_loc][0] = v2i{raw[0], raw[1]};
          a_frag[mi_loc][1] = v2i{raw[2], raw[3]};
        }

        // 确保本 chunk 的 DS 矩阵读取（含首轮 B_frag）完成后，再写 M0 并发射 v_mmac_scale
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        // 3. 在全部 ds_read_matrix_* 之后设置 wave 私有 M0（因 ds_read_matrix_* 会覆写 M0）。
        HCU_SET_SCALE_M0(my_row_a, my_row_b);

        // 4. 原生 gfx946 v_mmac_scale_f32_16x16x32_fp8_fp8（lit=true -> 每 lane 持有连续 4 列）。
        //    当 K 尾块 <= 32 时跳过 ks=1（CTA-uniform 分支），既防 NN 未加载半块残留 NaN 污染，
        //    又省去尾拍一半无效 MMAC。
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          if (ks == 0 || (k_cur + 32 < k_end)) {
            #pragma unroll
            for (int mi_loc = 0; mi_loc < M_CHUNK; ++mi_loc) {
              const int mi = mc * M_CHUNK + mi_loc;
              #pragma unroll
              for (int ni = 0; ni < WN_TILES; ++ni) {
                acc[mi][ni] = HCU_MMAC_SCALE_F32_FP8_FP8(
                    a_frag[mi_loc][ks], b_frag[ni][ks], acc[mi][ni], true, false, mi, 0);
              }
            }
          }
        }
      }

      // 仅在 DS 读取与 v_mmac_scale 全部完成后释放 EMPTY 屏障，确保 LDS 与 Scale Buffer 均无竞争
      __builtin_hcu_s_abarrier_arrive(STAGES + stage);
    }

    // 5. Epilogue：lit=true, lts=false => row = lane & 15, col = (lane >> 4) * 4 + [0..3]。
    const int r_lane = lane & 15;
    const int c_lane = (lane >> 4) * 4;
    #pragma unroll
    for (int mi = 0; mi < WM_TILES; ++mi) {
      const int64_t gm = m0 + wm_base + mi * 16 + r_lane;
      #pragma unroll
      for (int ni = 0; ni < WN_TILES; ++ni) {
        const int64_t gn = n0 + wn_base + ni * 16 + c_lane;
        store_tile_epilogue<OutT>(acc[mi][ni], D_b, ws_b, bias_row, gm, gn, M, N, split_k);
      }
    }
  }

  // 释放全部 2*STAGES 个硬件 Abarrier 槽位（Abarrier编程指南.pdf P2；加 s_nop 0 满足
  // ISA v1.8 §4.2.5 S_EBARRIER_SYNC -> S_ENDPGM 1-cycle hazard）。
  __builtin_hcu_s_ebarrier_sync(0);
  if (wave_id == 0) {
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      __builtin_hcu_s_abarrier_inv(s);
      __builtin_hcu_s_abarrier_inv(STAGES + s);
    }
  }
  asm volatile("s_nop 0" ::: "memory");
}

// ----------------------------------------------------------------------------
// Family 2: 轻量 4-Wave Decode / 短 K 内核（非 WASP，2-Stage LDS，低延迟 + Split-K）
// - 覆盖 M <= 64（Decode M=1..16、中小 batch 32..64）及短 K <= 256 投影（如 K=128/256）。
// - Scale Buffer 精确分配 4 slots（64 rows = 1024B），使单 CU 在 Scale Buffer 维度
//   可并发驻留多达 8 个 CTA，最大化小 batch 延迟隐藏。
// ----------------------------------------------------------------------------
template <
    typename OutT,
    int BM,
    int BN,
    int BK,
    int SCALE_BLK,
    bool B_TRANSPOSED>
__global__ __launch_bounds__(256)
__attribute__((hcu_scale_buffer_size(4)))
void fp8_blockwise_ue8m0_lite_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    const OutT* __restrict__ bias,
    OutT* __restrict__ D,
    float* __restrict__ workspace,
    int64_t M,
    int64_t N,
    int64_t K,
    int split_k,
    bool bias_batched) {
  static_assert(BK == 64, "BK must be 64");
  static_assert(SCALE_BLK == 128 || SCALE_BLK == 64, "SCALE_BLK must be 128 or 64");
  static_assert(BM == 16 || BM == 32 || BM == 64, "Lite BM must be 16, 32, or 64");
  static_assert(BN == 64 || BN == 128, "Lite BN must be 64 or 128");

  constexpr int M_WAVES = (BM >= 32) ? 2 : 1;
  constexpr int N_WAVES = 4 / M_WAVES;
  constexpr int WM = BM / M_WAVES;          // 16 or 32
  constexpr int WN = BN / N_WAVES;          // 16 or 64
  constexpr int WM_TILES = WM / 16;         // 1 or 2
  constexpr int WN_TILES = WN / 16;         // 1 or 4
  constexpr int K_STEPS_PER_SBLK = SCALE_BLK / BK;
  constexpr int NUM_B_BLKS = (BN + SCALE_BLK - 1) / SCALE_BLK;  // 1 or 2
  constexpr int A_STAGE_BYTES = BM * BK;
  constexpr int B_STAGE_BYTES = BN * BK;
  constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
  constexpr int SCALE_LDS_STRIDE = 256;
  constexpr int A_SBUF_ROWS = M_WAVES * 8;
  constexpr int B_SBUF_ROWS = NUM_B_BLKS * 8;
  constexpr int STAGE_SBUF_ROWS = A_SBUF_ROWS + B_SBUF_ROWS;
  static_assert(2 * STAGE_SBUF_ROWS <= 4 * 16,
                "Lite Scale Buffer usage exceeds 4 slots (64 rows)");

  __shared__ __align__(256) uint8_t smem[2 * STAGE_BYTES + 2 * SCALE_LDS_STRIDE];

  const int64_t batch_idx = static_cast<int64_t>(blockIdx.z) / split_k;
  const int split_idx = static_cast<int>(blockIdx.z) % split_k;
  const int64_t num_k_blks = (K + SCALE_BLK - 1) / SCALE_BLK;
  const int64_t num_n_blks = (N + SCALE_BLK - 1) / SCALE_BLK;
  const int64_t k_blks_per_split = (num_k_blks + split_k - 1) / split_k;
  const int64_t k_start = static_cast<int64_t>(split_idx) * k_blks_per_split * SCALE_BLK;
  const int64_t k_end = dmin64(K, static_cast<int64_t>(split_idx + 1) * k_blks_per_split * SCALE_BLK);
  if (k_start >= k_end) {
    return;
  }

  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * BN;

  const uint8_t* A_b = A + batch_idx * M * K;
  const uint8_t* B_b = B + batch_idx * K * N;
  const float* sa_b = scale_a + batch_idx * M * num_k_blks;
  const float* sb_b = scale_b + batch_idx * num_k_blks * num_n_blks;
  OutT* D_b = D + batch_idx * M * N;
  float* ws_b = (split_k > 1) ? (workspace + (batch_idx * split_k + split_idx) * M * N) : nullptr;
  const OutT* bias_row = (bias == nullptr) ? nullptr : (bias_batched ? (bias + batch_idx * N) : bias);

  const int wave_id = __builtin_amdgcn_readfirstlane(__builtin_hcu_get_wave_id());
  const int lane = threadIdx.x & 63;
  const int m_wave_id = wave_id / N_WAVES;
  const int n_wave_id = wave_id % N_WAVES;
  const int wm_base = m_wave_id * WM;
  const int wn_base = n_wave_id * WN;
  const int my_b_blk_local = wn_base / SCALE_BLK;

  v4f acc[WM_TILES][WN_TILES];
  #pragma unroll
  for (int mi = 0; mi < WM_TILES; ++mi) {
    #pragma unroll
    for (int ni = 0; ni < WN_TILES; ++ni) {
      acc[mi][ni] = v4f{0.f, 0.f, 0.f, 0.f};
    }
  }

  const int num_k_steps = static_cast<int>((k_end - k_start + BK - 1) / BK);

  for (int k_step = 0; k_step < num_k_steps; ++k_step) {
    const int stage = k_step & 1;
    const int64_t k_cur = k_start + static_cast<int64_t>(k_step) * BK;
    const int valid_k_64 = static_cast<int>(dmin64(BK, k_end - k_cur));
    const int k_filter_64 = BK - valid_k_64;

    uint8_t* a_lds = smem + stage * STAGE_BYTES;
    uint8_t* b_lds = a_lds + A_STAGE_BYTES;

    // Cooperative 4-wave MLS load (non-bps -> standard vmcnt(0) + ebarrier for ultra-low latency).
    if constexpr (BM == 16) {
      if (wave_id == 0) {
        const int valid_m = static_cast<int>(dmax64(0, dmin64(16, M - m0)));
        if (valid_m > 0) {
          v4i srsrc_a = make_mls_srsrc_uniform(
              A_b + m0 * K + k_cur, static_cast<int>(K), k_filter_64, 16 - valid_m);
          __builtin_hcu_matrix_load_64x16_b8(
              srsrc_a, to_lds_i8(a_lds), 0, true, false, false, false, false);
        }
      }
    } else {
      constexpr int A_BLOCKS = BM / 32;
      #pragma unroll
      for (int ab = wave_id; ab < A_BLOCKS; ab += 4) {
        const int64_t row_a = m0 + ab * 32;
        const int valid_m = static_cast<int>(dmax64(0, dmin64(32, M - row_a)));
        if (valid_m > 0) {
          v4i srsrc_a = make_mls_srsrc_uniform(
              A_b + row_a * K + k_cur, static_cast<int>(K), k_filter_64, 32 - valid_m);
          __builtin_hcu_matrix_load_64x32_b8(
              srsrc_a, to_lds_i8(a_lds + ab * 2048), 0, true, false, false, false, false);
        }
      }
    }

    if constexpr (B_TRANSPOSED) {
      constexpr int B_BLOCKS = BN / 32;
      #pragma unroll
      for (int bb = wave_id; bb < B_BLOCKS; bb += 4) {
        const int64_t row_b = n0 + bb * 32;
        const int valid_n = static_cast<int>(dmax64(0, dmin64(32, N - row_b)));
        if (valid_n > 0) {
          v4i srsrc_b = make_mls_srsrc_uniform(
              B_b + row_b * K + k_cur, static_cast<int>(K), k_filter_64, 32 - valid_n);
          __builtin_hcu_matrix_load_64x32_b8(
              srsrc_b, to_lds_i8(b_lds + bb * 2048), 0, true, false, false, false, false);
        }
      }
    } else {
      constexpr int B_N64 = BN / 64;
      constexpr int B_BLOCKS = B_N64 * 2;
      #pragma unroll
      for (int bb = wave_id; bb < B_BLOCKS; bb += 4) {
        const int n64_idx = bb / 2;
        const int k32_idx = bb & 1;
        const int64_t col_b = n0 + n64_idx * 64;
        const int64_t row_k = k_cur + k32_idx * 32;
        const int valid_n = static_cast<int>(dmax64(0, dmin64(64, N - col_b)));
        const int valid_k = static_cast<int>(dmax64(0, dmin64(32, k_end - row_k)));
        if (valid_n > 0 && valid_k > 0) {
          v4i srsrc_b = make_mls_srsrc_uniform(
              B_b + row_k * N + col_b, static_cast<int>(N), 64 - valid_n, 32 - valid_k);
          __builtin_hcu_matrix_load_64x32_b8(
              srsrc_b, to_lds_i8(b_lds + bb * 2048), 0, true, false, false, false, false);
        }
      }
    }

    if (wave_id == 0 && (k_step % K_STEPS_PER_SBLK) == 0) {
      const int64_t k_blk = dmin64(k_cur / SCALE_BLK, num_k_blks - 1);
      const int sb_slot = static_cast<int>(k_blk & 1);
      uint8_t* scale_lds = smem + 2 * STAGE_BYTES + sb_slot * SCALE_LDS_STRIDE;

      if (lane < BM) {
        const int64_t gm = dmin64(m0 + lane, M - 1);
        scale_lds[lane] = fp32_pow2_to_e8m0(sa_b[gm * num_k_blks + k_blk]);
      }
      if (lane < 16) {
        #pragma unroll
        for (int bi = 0; bi < NUM_B_BLKS; ++bi) {
          const int64_t n_blk = dmin64((n0 + bi * SCALE_BLK) / SCALE_BLK, num_n_blks - 1);
          const float sb_val = B_TRANSPOSED
                                   ? sb_b[n_blk * num_k_blks + k_blk]
                                   : sb_b[k_blk * num_n_blks + n_blk];
          scale_lds[BM + bi * 16 + lane] = fp32_pow2_to_e8m0(sb_val);
        }
      }
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      if (lane < 16) {
        const int base_row = sb_slot * STAGE_SBUF_ROWS;
        #pragma unroll
        for (int mw = 0; mw < M_WAVES; ++mw) {
          #pragma unroll
          for (int r16 = 0; r16 < WM_TILES; ++r16) {
            lds_i32_ptr src_a = to_lds_i32(
                scale_lds + mw * WM + r16 * 16 + lane);
            __builtin_hcu_ds_scale_copy_ds2buf(src_a, base_row + mw * 8 + r16, 0, 0, 0);
          }
        }
        #pragma unroll
        for (int bi = 0; bi < NUM_B_BLKS; ++bi) {
          lds_i32_ptr src_b = to_lds_i32(
              scale_lds + BM + bi * 16 + lane);
          __builtin_hcu_ds_scale_copy_ds2buf(
              src_b, base_row + A_SBUF_ROWS + bi * 8, 0, 0, 0);
        }
      }
    }

    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __builtin_hcu_s_ebarrier_sync(0);

    v2i a_frag[WM_TILES][2];
    v2i b_frag[WN_TILES][2];

    #pragma unroll
    for (int mi = 0; mi < WM_TILES; ++mi) {
      const int a_row = wm_base + mi * 16;
      const int a_off = (BM == 16) ? 0 : ((a_row / 32) * 2048 + ((a_row % 32) / 16) * 1024);
      v4i raw = __builtin_hcu_ds_read_matrix_trans_format_u8(
          to_lds_i32(a_lds + a_off), 0, 3, 1, 0);
      a_frag[mi][0] = v2i{raw[0], raw[1]};
      a_frag[mi][1] = v2i{raw[2], raw[3]};
    }

    if constexpr (B_TRANSPOSED) {
      #pragma unroll
      for (int ni = 0; ni < WN_TILES; ++ni) {
        const int b_row = wn_base + ni * 16;
        const int b_off = (b_row / 32) * 2048 + ((b_row % 32) / 16) * 1024;
        v4i raw = __builtin_hcu_ds_read_matrix_trans_format_u8(
            to_lds_i32(b_lds + b_off), 0, 3, 1, 0);
        b_frag[ni][0] = v2i{raw[0], raw[1]};
        b_frag[ni][1] = v2i{raw[2], raw[3]};
      }
    } else {
      if constexpr (WN_TILES == 1) {
        // WN == 16（16x64 档位）：读覆盖 [wn_base & ~31, +32) 的 32x32 块（后半 32 列偏移 +32B），
        // 再按 half_sel 选低/高 16 列。
        const int half_sel = (wn_base >> 4) & 1;
        const int b_col32 = wn_base & ~31;
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          const int b_off = ((b_col32 / 64) * 2 + ks) * 2048 + ((b_col32 % 64) / 32) * 32;
          v4i raw = __builtin_hcu_ds_read_matrix_format_u8(
              to_lds_i32(b_lds + b_off), 0, 2, 2, 0);
          b_frag[0][ks] = (half_sel == 0) ? v2i{raw[0], raw[1]} : v2i{raw[2], raw[3]};
        }
      } else {
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          #pragma unroll
          for (int ni2 = 0; ni2 < WN_TILES / 2; ++ni2) {
            const int b_col = wn_base + ni2 * 32;
            const int b_off = ((b_col / 64) * 2 + ks) * 2048 + ((b_col % 64) / 32) * 32;
            v4i raw = __builtin_hcu_ds_read_matrix_format_u8(
                to_lds_i32(b_lds + b_off), 0, 2, 2, 0);
            b_frag[2 * ni2 + 0][ks] = v2i{raw[0], raw[1]};
            b_frag[2 * ni2 + 1][ks] = v2i{raw[2], raw[3]};
          }
        }
      }
    }

    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_hcu_s_ebarrier_sync(0);

    const int sb_slot = static_cast<int>((k_cur / SCALE_BLK) & 1);
    const int my_row_a = sb_slot * STAGE_SBUF_ROWS + m_wave_id * 8;
    const int my_row_b = sb_slot * STAGE_SBUF_ROWS + A_SBUF_ROWS + my_b_blk_local * 8;
    HCU_SET_SCALE_M0(my_row_a, my_row_b);

    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      if (ks == 0 || (k_cur + 32 < k_end)) {
        #pragma unroll
        for (int mi = 0; mi < WM_TILES; ++mi) {
          #pragma unroll
          for (int ni = 0; ni < WN_TILES; ++ni) {
            acc[mi][ni] = HCU_MMAC_SCALE_F32_FP8_FP8(
                a_frag[mi][ks], b_frag[ni][ks], acc[mi][ni], true, false, mi, 0);
          }
        }
      }
    }
  }

  const int r_lane = lane & 15;
  const int c_lane = (lane >> 4) * 4;
  #pragma unroll
  for (int mi = 0; mi < WM_TILES; ++mi) {
    const int64_t gm = m0 + wm_base + mi * 16 + r_lane;
    #pragma unroll
    for (int ni = 0; ni < WN_TILES; ++ni) {
      const int64_t gn = n0 + wn_base + ni * 16 + c_lane;
      store_tile_epilogue<OutT>(acc[mi][ni], D_b, ws_b, bias_row, gm, gn, M, N, split_k);
    }
  }
}

#else

template <
    typename OutT, int BM, int BN, int BK, int STAGES, int SCALE_BLK,
    bool B_TRANSPOSED, int PRODUCER_WAVES, int CONSUMER_WAVES>
__global__ void fp8_blockwise_ue8m0_wasp_kernel(
    const uint8_t*, const uint8_t*, const float*, const float*,
    const OutT*, OutT*, float*, int64_t, int64_t, int64_t, int, bool) {}

template <typename OutT, int BM, int BN, int BK, int SCALE_BLK, bool B_TRANSPOSED>
__global__ void fp8_blockwise_ue8m0_lite_kernel(
    const uint8_t*, const uint8_t*, const float*, const float*,
    const OutT*, OutT*, float*, int64_t, int64_t, int64_t, int, bool) {}

#endif  // defined(__gfx946__)

template <typename OutT>
__global__ void fp8_blockwise_splitk_reduce_kernel(
    const float* __restrict__ workspace,
    const OutT* __restrict__ bias,
    OutT* __restrict__ out,
    int64_t batch,
    int64_t M,
    int64_t N,
    int split_k,
    bool bias_batched) {
  const int64_t mn = M * N;
  const int64_t total_vec = (batch * mn) / 4;
  const int64_t idx_vec = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx_vec >= total_vec) {
    return;
  }
  const int64_t elem0 = idx_vec * 4;
  const int64_t b = elem0 / mn;
  const int64_t rem = elem0 % mn;
  const int64_t col = rem % N;

  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  const float* ws_base = workspace + b * split_k * mn + rem;
  for (int sk = 0; sk < split_k; ++sk) {
    float4 v = *reinterpret_cast<const float4*>(ws_base + sk * mn);
    s0 += v.x;
    s1 += v.y;
    s2 += v.z;
    s3 += v.w;
  }
  if (bias != nullptr) {
    const OutT* brow = bias_batched ? (bias + b * N) : bias;
    if constexpr (std::is_same_v<OutT, bhalf_t>) {
      s0 += __bfloat162float(brow[col + 0]);
      s1 += __bfloat162float(brow[col + 1]);
      s2 += __bfloat162float(brow[col + 2]);
      s3 += __bfloat162float(brow[col + 3]);
    } else {
      s0 += __half2float(brow[col + 0]);
      s1 += __half2float(brow[col + 1]);
      s2 += __half2float(brow[col + 2]);
      s3 += __half2float(brow[col + 3]);
    }
  }
  OutT* dst = out + elem0;
  if constexpr (std::is_same_v<OutT, bhalf_t>) {
    dst[0] = __float2bfloat16(s0);
    dst[1] = __float2bfloat16(s1);
    dst[2] = __float2bfloat16(s2);
    dst[3] = __float2bfloat16(s3);
  } else {
    dst[0] = __float2half(s0);
    dst[1] = __float2half(s1);
    dst[2] = __float2half(s2);
    dst[3] = __float2half(s3);
  }
}

static int choose_split_k(int64_t batch, int64_t M, int64_t N, int64_t K,
                          int bm, int bn, int scale_blk, int num_cus) {
  const int64_t ctas = batch * ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
  const int64_t num_k_blks = (K + scale_blk - 1) / scale_blk;
  if (ctas >= num_cus || num_k_blks < 4) {
    return 1;
  }
  int target = static_cast<int>((num_cus + ctas - 1) / ctas);
  int sk = 1;
  while (sk * 2 <= target && sk * 2 <= 16 && (num_k_blks / (sk * 2)) >= 2) {
    sk *= 2;
  }
  return sk;
}

template <typename OutT, int SCALE_BLK, bool B_TRANSPOSED>
static void launch_typed(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& out,
    int64_t M, int64_t N, int64_t K, int64_t batch,
    int num_cus,
    hipStream_t stream) {
  const uint8_t* a_ptr = reinterpret_cast<const uint8_t*>(a.data_ptr());
  const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(b.data_ptr());
  const float* sa_ptr = scale_a.data_ptr<float>();
  const float* sb_ptr = scale_b.data_ptr<float>();
  OutT* d_ptr = reinterpret_cast<OutT*>(out.data_ptr());
  const OutT* bias_ptr =
      (bias.has_value() && bias.value().defined())
          ? reinterpret_cast<const OutT*>(bias.value().data_ptr())
          : nullptr;
  const bool bias_batched =
      (bias_ptr != nullptr) && (bias.value().numel() == batch * N) && (batch > 1);

  // Select tile regime for gfx946 (192 CUs) & gfx938-aligned Dense GEMM shapes.
  int tier = 0;  // 0: A1-Large 256x256, 1: A1-Std 128x256, 2: A2 128x128, 3: Lite 64x128, 4: Lite 32x128, 5: Lite 16x64
  if (M <= 16) {
    tier = 5;
  } else if (M <= 32) {
    tier = 4;
  } else if (M <= 64 || K <= 256) {
    tier = 3;
  } else if (M >= 4096 && N >= 2048 && (batch * ((M + 255) / 256) * ((N + 255) / 256)) >= num_cus) {
    tier = 0;
  } else if (M >= 256 && N >= 512) {
    tier = 1;
  } else {
    tier = 2;
  }

  constexpr int kTileM[] = {256, 128, 128, 64, 32, 16};
  constexpr int kTileN[] = {256, 256, 128, 128, 128, 64};
  const int bm = kTileM[tier];
  const int bn = kTileN[tier];
  const int split_k = (tier >= 2) ? choose_split_k(batch, M, N, K, bm, bn, SCALE_BLK, num_cus) : 1;

  // Optional hot-plug ASM dispatch when hsa/gfx946/gemm_w8a8_fp8_block/*.{co,so,hsaco} exists.
  const std::string stem =
      std::string("gemm_w8a8_fp8_ue8m0_blk") + std::to_string(SCALE_BLK) + "_" +
      std::to_string(bm) + "x" + std::to_string(bn) + "x64_" +
      (B_TRANSPOSED ? "NT" : "NN") + "_" +
      (std::is_same_v<OutT, bhalf_t> ? "BF16" : "FP16");
  if (AiterAsmKernel* asm_k = try_get_asm_kernel(stem, stem)) {
    Fp8BlockwiseUe8m0AsmArgs args{
        d_ptr, a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr,
        M, N, K, batch, split_k, SCALE_BLK, B_TRANSPOSED ? 1 : 0,
        bias_ptr != nullptr ? 1 : 0, bias_batched ? 1 : 0, 0};
    size_t args_size = sizeof(args);
    const size_t bdx = (tier <= 1) ? 768 : ((tier == 2) ? 512 : 256);
    asm_k->launch_kernel_ext({
        &args, &args_size,
        static_cast<size_t>((N + bn - 1) / bn),
        static_cast<size_t>((M + bm - 1) / bm),
        static_cast<size_t>(batch * split_k),
        bdx, 1, 1, stream});
    return;
  }

  torch::Tensor ws;
  float* ws_ptr = nullptr;
  if (split_k > 1) {
    ws = at::empty({batch, split_k, M, N}, a.options().dtype(at::kFloat));
    ws_ptr = ws.data_ptr<float>();
  }

  dim3 grid(static_cast<unsigned>((N + bn - 1) / bn),
            static_cast<unsigned>((M + bm - 1) / bm),
            static_cast<unsigned>(batch * split_k));

  if (tier == 0) {
    fp8_blockwise_ue8m0_wasp_kernel<OutT, 256, 256, 64, 3, SCALE_BLK, B_TRANSPOSED, 4, 8>
        <<<grid, 768, 0, stream>>>(
            a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, d_ptr, ws_ptr, M, N, K, split_k, bias_batched);
  } else if (tier == 1) {
    fp8_blockwise_ue8m0_wasp_kernel<OutT, 128, 256, 64, 4, SCALE_BLK, B_TRANSPOSED, 4, 8>
        <<<grid, 768, 0, stream>>>(
            a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, d_ptr, ws_ptr, M, N, K, split_k, bias_batched);
  } else if (tier == 2) {
    fp8_blockwise_ue8m0_wasp_kernel<OutT, 128, 128, 64, 4, SCALE_BLK, B_TRANSPOSED, 4, 4>
        <<<grid, 512, 0, stream>>>(
            a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, d_ptr, ws_ptr, M, N, K, split_k, bias_batched);
  } else if (tier == 3) {
    fp8_blockwise_ue8m0_lite_kernel<OutT, 64, 128, 64, SCALE_BLK, B_TRANSPOSED>
        <<<grid, 256, 0, stream>>>(
            a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, d_ptr, ws_ptr, M, N, K, split_k, bias_batched);
  } else if (tier == 4) {
    fp8_blockwise_ue8m0_lite_kernel<OutT, 32, 128, 64, SCALE_BLK, B_TRANSPOSED>
        <<<grid, 256, 0, stream>>>(
            a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, d_ptr, ws_ptr, M, N, K, split_k, bias_batched);
  } else {
    fp8_blockwise_ue8m0_lite_kernel<OutT, 16, 64, 64, SCALE_BLK, B_TRANSPOSED>
        <<<grid, 256, 0, stream>>>(
            a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, d_ptr, ws_ptr, M, N, K, split_k, bias_batched);
  }

  if (split_k > 1) {
    const int64_t total_vec = (batch * M * N) / 4;
    const int threads = 256;
    const int blocks = static_cast<int>((total_vec + threads - 1) / threads);
    fp8_blockwise_splitk_reduce_kernel<OutT><<<blocks, threads, 0, stream>>>(
        ws_ptr, bias_ptr, d_ptr, batch, M, N, split_k, bias_batched);
  }
}

}  // namespace fp8_blockwise_946

torch::Tensor fp8_blockwise_gemm_ue8m0_gfx946(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    at::ScalarType out_dtype,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch,
    const std::string& transpose_flag,
    int block_size,
    const c10::optional<torch::Tensor>& bias) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

  int device = 0;
  hipGetDevice(&device);
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, device);
  TORCH_CHECK(std::strncmp(prop.gcnArchName, "gfx946", 6) == 0,
              "fp8_blockwise_gemm_ue8m0_gfx946 requires gfx946, got ", prop.gcnArchName);
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn,
              "fp8_blockwise_gemm_ue8m0_gfx946 requires float8_e4m3fn inputs");
  TORCH_CHECK(scale_a.scalar_type() == torch::kFloat32 && scale_b.scalar_type() == torch::kFloat32,
              "fp8_blockwise_gemm_ue8m0_gfx946 requires float32 UE8M0 scales");
  TORCH_CHECK(out_dtype == torch::kBFloat16 || out_dtype == torch::kHalf,
              "fp8_blockwise_gemm_ue8m0_gfx946 supports bfloat16 or float16 output");
  TORCH_CHECK(block_size == 128 || block_size == 64,
              "fp8_blockwise_gemm_ue8m0_gfx946 supports block_size 128 or 64");
  TORCH_CHECK(transpose_flag == "NN" || transpose_flag == "NT",
              "fp8_blockwise_gemm_ue8m0_gfx946 supports NN or NT layout");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() &&
                  scale_a.is_contiguous() && scale_b.is_contiguous(),
              "fp8_blockwise_gemm_ue8m0_gfx946 requires contiguous tensors");
  TORCH_CHECK(k % 16 == 0 && n % 16 == 0 &&
                  (reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0) &&
                  (reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0),
              "fp8_blockwise_gemm_ue8m0_gfx946 requires 16-byte alignment for K, N, and data pointers");
  TORCH_CHECK(batch >= 1, "batch must be >= 1");
  const int64_t k_blks = (k + block_size - 1) / block_size;
  const int64_t n_blks = (n + block_size - 1) / block_size;
  TORCH_CHECK(a.numel() == batch * m * k && b.numel() == batch * n * k,
              "fp8_blockwise_gemm_ue8m0_gfx946 tensor size mismatch for batch/m/n/k");
  TORCH_CHECK(scale_a.numel() == batch * m * k_blks &&
                  scale_b.numel() == batch * k_blks * n_blks,
              "fp8_blockwise_gemm_ue8m0_gfx946 scale size mismatch");

  c10::optional<torch::Tensor> bias_contig = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    const torch::Tensor& bt = bias.value();
    TORCH_CHECK(bt.scalar_type() == out_dtype,
                "fp8_blockwise_gemm_ue8m0_gfx946 bias dtype must match out_dtype");
    TORCH_CHECK(bt.numel() == n || bt.numel() == batch * n,
                "fp8_blockwise_gemm_ue8m0_gfx946 bias must have N or batch*N elements");
    bias_contig = bt.is_contiguous() ? bt : bt.contiguous();
  }

  torch::Tensor out = at::empty({batch, m, n}, a.options().dtype(out_dtype));
  if (m == 0 || n == 0 || k == 0) {
    out.zero_();
    return out;
  }

  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  const bool b_transposed = (transpose_flag == "NT");
  const int num_cus = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 192;

  auto dispatch_blk = [&](auto out_tag) {
    using OutT = decltype(out_tag);
    if (block_size == 128) {
      if (b_transposed) {
        fp8_blockwise_946::launch_typed<OutT, 128, true>(
            a, b, scale_a, scale_b, bias_contig, out, m, n, k, batch, num_cus, stream);
      } else {
        fp8_blockwise_946::launch_typed<OutT, 128, false>(
            a, b, scale_a, scale_b, bias_contig, out, m, n, k, batch, num_cus, stream);
      }
    } else {
      if (b_transposed) {
        fp8_blockwise_946::launch_typed<OutT, 64, true>(
            a, b, scale_a, scale_b, bias_contig, out, m, n, k, batch, num_cus, stream);
      } else {
        fp8_blockwise_946::launch_typed<OutT, 64, false>(
            a, b, scale_a, scale_b, bias_contig, out, m, n, k, batch, num_cus, stream);
      }
    }
  };

  if (out_dtype == torch::kBFloat16) {
    dispatch_blk(bhalf_t{});
  } else {
    dispatch_blk(half_t{});
  }
  return out;
}

} // namespace native
} // namespace at