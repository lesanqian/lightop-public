// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Public interfaces implemented by csrc/moe_gemm/{w16a16,w4a8,w8a8}/*.cu,
// csrc/kernels/moe_gate/*.cu, and root MoE kernels.
// Keep signatures and Python defaults aligned with csrc/pybind/export.cpp.
#include <torch/extension.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace at::native {

Tensor moe_gemm_w8a8(torch::Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_marlin_w8a8(torch::Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_marlin_w8a8_fp8(torch::Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_marlin_w8a8_blockwise_fp8(torch::Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_w4a8(torch::Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_w16a16(torch::Tensor input, Tensor b_qweight, Tensor output, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);

void moe_align_block_size(
    torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    const torch::optional<torch::Tensor>& expert_map = torch::nullopt,
    const torch::optional<torch::Tensor>& expert_mask = torch::nullopt,
    const torch::optional<torch::Tensor>& num_local_tokens = torch::nullopt,
    bool Is_EP = false, bool Is_fuse_fill = true);

void moe_sum(
    torch::Tensor& input,
    torch::Tensor& output,
    const torch::optional<torch::Tensor>& bias = torch::nullopt,
    const torch::optional<torch::Tensor>& expert_mask = torch::nullopt,
    const torch::optional<torch::Tensor>& num_local_tokens = torch::nullopt,
    const float factor = 1.f, const int expect_m = -1);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> groupgemm_permute(
    const torch::Tensor& tokens, const torch::Tensor& routing_map);
torch::Tensor groupgemm_unpermute(
    const torch::Tensor& permuted_tokens, const torch::Tensor& sorted_indices,
    const std::vector<int64_t>& restore_shape, std::optional<torch::Tensor> probs,
    std::optional<torch::Tensor> routing_map, const torch::Tensor& expert_m_count,
    bool use_fp32);

std::vector<Tensor> moe_fused_gate(
    torch::Tensor& input, torch::Tensor& bias, int64_t num_expert_group,
    int64_t topk_group, int64_t topk, int64_t n_share_experts_fusion,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output,
    bool output_indices_int64);
std::vector<Tensor> moe_fuse_gate_per_gourp_topk(
    torch::Tensor& input, torch::Tensor& bias, int64_t num_expert_group,
    int64_t topk_per_group, int64_t num_pad_experts, int64_t num_shared_experts,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);
std::vector<at::Tensor> kimi_k2_moe_fused_gate(
    at::Tensor& input, at::Tensor& bias, int64_t topk, bool renormalize,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);
std::vector<Tensor> moe_fused_gate_sqrtsoftplus(
    torch::Tensor& input, torch::Tensor& bias, int64_t topk,
    int64_t num_fused_shared_experts, bool renormalize,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);
std::vector<Tensor> bf16_mega_gate_hcu_impl(
    const Tensor& x, const Tensor& weight, const torch::optional<Tensor>& optional_bias,
    int64_t topk, bool renormalize, double routed_scaling_factor, bool apply_scale,
    const torch::optional<std::vector<Tensor>>& out,
    const torch::optional<Tensor>& optional_mask);
std::tuple<Tensor, Tensor> bf16_mega_gate_hcu(
    const Tensor& x, const Tensor& weight, int64_t num_topk,
    bool use_shared_as_routed, int64_t num_shared_experts,
    double routed_scaling_factor, int64_t ep_rank, const std::string& scoring_func,
    const torch::optional<Tensor>& mask, const torch::optional<Tensor>& bias,
    const torch::optional<Tensor>& image_bias,
    const torch::optional<Tensor>& image_token_mask,
    const torch::optional<Tensor>& fix_routing_mask,
    const torch::optional<Tensor>& to_physical_map,
    const torch::optional<Tensor>& logical_count,
    const torch::optional<Tensor>& unmapped_topk_idx,
    const torch::optional<Tensor>& force_random,
    const torch::optional<std::tuple<Tensor, Tensor>>& out);
std::vector<Tensor> moe_fused_gate_sqrtsoftplus_v41_hcu(
    torch::Tensor& input, torch::Tensor& bias, int64_t topk,
    int64_t num_fused_shared_experts, bool renormalize,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);
std::vector<Tensor> moe_fused_gate_sqrtsoftplus_v41_hcu_fused(
    torch::Tensor& input, torch::Tensor& bias, int64_t topk,
    int64_t num_fused_shared_experts, bool renormalize,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output,
    torch::optional<torch::Tensor> num_token_non_padded);
std::vector<Tensor> moe_fused_gate_sqrtsoftplus_v41_hcu_deepep(
    torch::Tensor& input, torch::Tensor& bias, int64_t topk,
    int64_t num_fused_shared_experts, bool renormalize,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output,
    torch::optional<torch::Tensor> num_token_non_padded);

void moe_swiglu_dynamic_quant(
    torch::Tensor& scatter_tokens, torch::Tensor& smooth,
    torch::Tensor& experts_tokens_count, torch::Tensor& experts_tokens_start,
    torch::Tensor& output, torch::Tensor& scales, float beta);

void ep_fused_quant_scatter(
    torch::Tensor input, torch::Tensor topk_ids,
    torch::optional<torch::Tensor> expert_map_opt, torch::Tensor expert_num_tokens,
    torch::Tensor aq_out, torch::Tensor aq_scale_out, torch::Tensor m_indices,
    torch::Tensor inv_perm, int local_num_experts, int alignment);
void ep_scatter(
    torch::Tensor aq, torch::Tensor aq_scale, torch::Tensor topk_ids,
    torch::optional<torch::Tensor> expert_map_opt, torch::Tensor expert_num_tokens,
    torch::Tensor aq_out, torch::Tensor aq_scale_out,
    torch::optional<torch::Tensor> m_indices_opt, torch::Tensor inv_perm,
    int local_num_experts, int alignment);
void ep_gather(
    torch::Tensor& a, torch::Tensor& topk_ids, torch::Tensor& topk_weights,
    torch::Tensor& inv_perm, torch::optional<torch::Tensor> expert_map,
    torch::Tensor& output, bool specialize_k8 = false);
void ep_fused_smooth_quant_scatter(
    torch::Tensor input, torch::Tensor topk_ids,
    torch::optional<torch::Tensor> expert_map_opt, torch::Tensor expert_offsets,
    torch::Tensor smooth_scale, torch::Tensor aq_out, torch::Tensor aq_scale_out,
    torch::Tensor m_indices, torch::Tensor inv_perm,
    int local_num_experts, int alignment);
void ep_build_m_indices(
    torch::Tensor topk_ids, torch::Tensor m_indices,
    int local_num_experts, int alignment);
void ep_fused_fp8_quant_scatter(
    torch::Tensor input, torch::Tensor topk_ids,
    torch::optional<torch::Tensor> expert_map_opt, torch::Tensor expert_num_tokens,
    torch::Tensor aq_out, torch::Tensor aq_scale_out, torch::Tensor m_indices,
    torch::Tensor inv_perm, int local_num_experts, int alignment,
    int fp8type, bool fill_padded_m_indices);

Tensor fused_clamp_smooth(
    const Tensor& hidden, const Tensor& smooth,
    double min_limit, double max_limit);
std::tuple<Tensor, Tensor> fused_deepep_prepare(
    const Tensor& hidden, const Tensor& smooth, const Tensor& ids,
    double min_limit, double max_limit);

}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp MoE GEMM, gate, permute, and routing operators";

  m.def("moe_gemm_w8a8", &at::native::moe_gemm_w8a8, "moe_gemm_w8a8");
  m.def("moe_gemm_marlin_w8a8", &at::native::moe_gemm_marlin_w8a8, "moe_gemm_marlin_w8a8");
  m.def("moe_gemm_marlin_w8a8_fp8", &at::native::moe_gemm_marlin_w8a8_fp8, "moe_gemm_marlin_w8a8_fp8");
  m.def("moe_gemm_marlin_w8a8_blockwise_fp8", &at::native::moe_gemm_marlin_w8a8_blockwise_fp8, "moe_gemm_marlin_w8a8_blockwise_fp8");
  m.def("moe_gemm_w4a8", &at::native::moe_gemm_w4a8, "moe_gemm_w4a8");
  m.def("moe_gemm_w16a16", &at::native::moe_gemm_w16a16, "moe_gemm_w16a16");

  m.def("groupgemm_permute", &at::native::groupgemm_permute, "groupgemm_permute");
  m.def("groupgemm_unpermute", &at::native::groupgemm_unpermute, "groupgemm_unpermute");

  m.def(
      "moe_fused_gate",
      &at::native::moe_fused_gate,
      "moe_fused_gate",
      py::arg("input"),
      py::arg("bias"),
      py::arg("num_expert_group"),
      py::arg("topk_group"),
      py::arg("topk"),
      py::arg("n_share_experts_fusion"),
      py::arg("routed_scaling_factor"),
      py::arg("apply_routed_scaling_factor_on_output") = true,
      py::arg("output_indices_int64") = false);
  m.def(
      "moe_fuse_gate_per_gourp_topk",
      &at::native::moe_fuse_gate_per_gourp_topk,
      "SparkV1 per-group top-k fused MoE gate",
      py::arg("input"),
      py::arg("bias"),
      py::arg("num_expert_group"),
      py::arg("topk_per_group"),
      py::arg("num_pad_experts"),
      py::arg("num_shared_experts"),
      py::arg("routed_scaling_factor"),
      py::arg("apply_routed_scaling_factor_on_output") = true);
  m.def("kimi_k2_moe_fused_gate", &at::native::kimi_k2_moe_fused_gate, "kimi_k2_moe_fused_gate");
  m.def("moe_fused_gate_sqrtsoftplus", &at::native::moe_fused_gate_sqrtsoftplus, "moe_fused_gate_sqrtsoftplus");
  m.def("bf16_mega_gate_hcu", &at::native::bf16_mega_gate_hcu,
        py::arg("x"), py::arg("weight"), py::arg("num_topk"),
        py::arg("use_shared_as_routed"), py::arg("num_shared_experts"),
        py::arg("routed_scaling_factor"), py::arg("ep_rank"),
        py::arg("scoring_func") = "sqrtsoftplus", py::arg("mask") = py::none(),
        py::arg("bias") = py::none(), py::arg("image_bias") = py::none(),
        py::arg("image_token_mask") = py::none(), py::arg("fix_routing_mask") = py::none(),
        py::arg("to_physical_map") = py::none(), py::arg("logical_count") = py::none(),
        py::arg("unmapped_topk_idx") = py::none(), py::arg("force_random") = py::none(),
        py::arg("out") = py::none());
  m.attr("bf16_mega_gate_hcu_api_version") = 4;
  m.def("_bf16_mega_gate_hcu", &at::native::bf16_mega_gate_hcu_impl,
        "HCU BF16 fused router with per-stream counters and managed scratch",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("topk") = 6,
        py::arg("renormalize") = true, py::arg("routed_scaling_factor") = 1.0,
        py::arg("apply_scale") = false, py::arg("out") = py::none(),
        py::arg("mask") = py::none());
  m.def("moe_fused_gate_sqrtsoftplus_v41_hcu", &at::native::moe_fused_gate_sqrtsoftplus_v41_hcu, "moe_fused_gate_sqrtsoftplus_v41_hcu");
  m.def("moe_fused_gate_sqrtsoftplus_v41_hcu_fused",
        &at::native::moe_fused_gate_sqrtsoftplus_v41_hcu_fused,
        py::arg("input"), py::arg("bias"), py::arg("topk"),
        py::arg("num_fused_shared_experts"), py::arg("renormalize"),
        py::arg("routed_scaling_factor"),
        py::arg("apply_routed_scaling_factor_on_output"),
        py::arg("num_token_non_padded") = py::none());
  m.def("moe_fused_gate_sqrtsoftplus_v41_hcu_deepep",
        &at::native::moe_fused_gate_sqrtsoftplus_v41_hcu_deepep,
        py::arg("input"), py::arg("bias"), py::arg("topk"),
        py::arg("num_fused_shared_experts"), py::arg("renormalize"),
        py::arg("routed_scaling_factor"),
        py::arg("apply_routed_scaling_factor_on_output"),
        py::arg("num_token_non_padded") = py::none());

  m.def("moe_sum", &at::native::moe_sum, "moe_sum",
        py::arg("input"),
        py::arg("output"),
        py::arg("bias") = torch::nullopt,
        py::arg("expert_mask") = torch::nullopt,
        py::arg("num_local_tokens") = torch::nullopt,
        py::arg("factor") = 1.0f,
        py::arg("expect_m") = -1);
  m.def("moe_align_block_size", &at::native::moe_align_block_size, "moe_align_block_size",
        py::arg("topk_ids"),
        py::arg("num_experts"),
        py::arg("block_size"),
        py::arg("sorted_token_ids"),
        py::arg("experts_ids"),
        py::arg("num_tokens_post_pad"),
        py::arg("expert_map") = torch::nullopt,
        py::arg("expert_mask") = torch::nullopt,
        py::arg("num_local_tokens") = torch::nullopt,
        py::arg("Is_EP") = false,
        py::arg("Is_fuse_fill") = true);

  m.def("moe_swiglu_dynamic_quant", &at::native::moe_swiglu_dynamic_quant, "moe_swiglu_dynamic_quant");

  m.def("ep_fused_quant_scatter", &at::native::ep_fused_quant_scatter, "ep_fused_quant_scatter");
  m.def("ep_scatter", &at::native::ep_scatter, "ep_scatter");
  m.def("ep_gather", &at::native::ep_gather, "ep_gather",
        py::arg("a"), py::arg("topk_ids"), py::arg("topk_weights"),
        py::arg("inv_perm"), py::arg("expert_map"), py::arg("output"),
        py::arg("specialize_k8") = false);
  m.def("ep_fused_smooth_quant_scatter", &at::native::ep_fused_smooth_quant_scatter, "ep_fused_smooth_quant_scatter");
  m.def("ep_build_m_indices", &at::native::ep_build_m_indices, "ep_build_m_indices");
  m.def("ep_fused_fp8_quant_scatter", &at::native::ep_fused_fp8_quant_scatter, "ep_fused_fp8_quant_scatter");

  m.def("fused_clamp_smooth", &at::native::fused_clamp_smooth, "fused_clamp_smooth");
  m.def("fused_deepep_prepare", &at::native::fused_deepep_prepare, "fused_deepep_prepare");
}
