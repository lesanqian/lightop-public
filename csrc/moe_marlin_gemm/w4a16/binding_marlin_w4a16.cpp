// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Public interfaces implemented by csrc/moe_marlin_gemm/w4a16/*.cu.
// Keep signatures and Python defaults aligned with csrc/pybind/export.cpp.
#include <torch/extension.h>

#include <optional>

namespace at::native {

Tensor moe_marlin_w4a16(
    torch::Tensor& a,
    std::optional<torch::Tensor> const& c_or_none,
    torch::Tensor& b_q_weight,
    torch::Tensor& b_scale_zeros,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    int64_t thread_n = -1,
    int64_t thread_k = -1,
    int64_t blocks_per_cu = 1);

Tensor awq_marlin_repack(
    torch::Tensor& b_q_weight,
    int64_t size_k,
    int64_t size_n,
    int64_t num_bits);

}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp Marlin W4A16 MoE operators";
  m.def("moe_marlin_w4a16", &at::native::moe_marlin_w4a16, "moe_marlin_w4a16");
  m.def("awq_marlin_repack", &at::native::awq_marlin_repack, "awq_marlin_repack");
}
