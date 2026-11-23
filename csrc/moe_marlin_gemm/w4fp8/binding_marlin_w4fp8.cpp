// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Public interfaces implemented by csrc/moe_marlin_gemm/w4fp8/*.cu.
// Keep signatures and Python defaults aligned with csrc/pybind/export.cpp.
#include <torch/extension.h>

#include <optional>

namespace at::native {

Tensor moe_marlin_w4fp8(
    torch::Tensor& a_q,
    torch::Tensor& a_scales,
    torch::Tensor& c,
    torch::Tensor& b_q_weight,
    torch::Tensor& b_scales,
    torch::Tensor& workspace,
    torch::Tensor& global_reduce_buffer,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded,
    std::optional<torch::Tensor> topk_weights,
    int64_t moe_block_size,
    int64_t top_k,
    bool is_ep,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    int64_t thread_n,
    int64_t thread_k,
    int64_t blocks_per_cu,
    bool is_reduce);

}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp Marlin W4FP8 MoE operators";
  m.def("moe_marlin_w4fp8", &at::native::moe_marlin_w4fp8, "moe_marlin_w4fp8");
}
