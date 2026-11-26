// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for the ASM GEMM/MoE family implemented by csrc/asm/*.cu.

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/macros/Macros.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace at::native {
std::vector<torch::Tensor> grougemm_w8a8_pertensor_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, std::vector<torch::Tensor>& As_list, std::vector<torch::Tensor>& Bs_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
std::vector<torch::Tensor> grougemm_bf16_fp16_nt_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
std::vector<torch::Tensor> grougemm_bf16_fp16_tn_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
std::vector<torch::Tensor> grougemm_bf16_fp16_nn_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
Tensor m_grouped_w8a8_gemm_nt_contig_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor m_indices, int mode);
Tensor m_grouped_w16a16_gemm_nt_contig_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor m_indices, int mode);
Tensor m_grouped_w8a8_gemm_nt_nopad_contig_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor m_indices, Tensor token_per_expert);
Tensor gemm_w8a8_smooth_asm(Tensor &A, Tensor &B, Tensor &As, Tensor &Bs, const std::optional<Tensor> bias, torch::Dtype output_dtype, int M, int N, int K, bool isTunning_slide_block, int tunning_slide_block, bool input_scale_first);
Tensor moe_w16a16_asm(Tensor input,Tensor b_qweight,Tensor output,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int top_k, int mode, int delta);
Tensor moe_marlin_w16a16_asm(Tensor input,Tensor b_qweight,Tensor output,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int top_k, int mode, int delta);
Tensor gemm_w8a8_asm(Tensor &A, Tensor &B, Tensor &As, Tensor &Bs, c10::optional<Tensor> &C, c10::optional<Tensor> &D_, c10::optional<Tensor> bias, torch::Dtype output_dtype, int M, int N, int K, int gradType = 0);
Tensor moe_w8a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode, int delta);
Tensor moe_marlin_w8a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode, int delta);
Tensor moe_marlin_w8a8_fp8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode, int delta);
Tensor moe_w4a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode);
Tensor moe_w4a8_marlin_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode);
Tensor moe_groupgemm_w4a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor problem_size, Tensor expert_offsets, int mode);
Tensor moe_groupgemm_marlin_w4a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor problem_size, Tensor expert_offsets, int mode);
Tensor m_grouped_marlin_w4a8_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor masked_m, int expected_m_per_group, int mode);
Tensor m_grouped_marlin_w8a8_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor masked_m, int expected_m_per_group, int mode);
Tensor m_grouped_marlin_fp8_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor masked_m, int expected_m_per_group, int mode);
Tensor m_grouped_bf16_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor masked_m, int expected_m_per_group, int mode);
Tensor groupgemm(Tensor &a, Tensor &b, Tensor &batch_sizes, bool trans_b);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grougemm_w8a8_pertensor_asm", &at::native::grougemm_w8a8_pertensor_asm, "grougemm_w8a8_pertensor_asm");
  m.def("grougemm_bf16_fp16_nt_asm", &at::native::grougemm_bf16_fp16_nt_asm, "grougemm_bf16_fp16_nt_asm");
  m.def("grougemm_bf16_fp16_tn_asm", &at::native::grougemm_bf16_fp16_tn_asm, "grougemm_bf16_fp16_tn_asm");
  m.def("grougemm_bf16_fp16_nn_asm", &at::native::grougemm_bf16_fp16_nn_asm, "grougemm_bf16_fp16_nn_asm");
  m.def("m_grouped_w8a8_gemm_nt_contig_asm", &at::native::m_grouped_w8a8_gemm_nt_contig_asm, "m_grouped_w8a8_gemm_nt_contig_asm");
  m.def("m_grouped_w16a16_gemm_nt_contig_asm",&at::native::m_grouped_w16a16_gemm_nt_contig_asm,"m_grouped_w16a16_gemm_nt_contig_asm");
  m.def("m_grouped_w8a8_gemm_nt_nopad_contig_asm",&at::native::m_grouped_w8a8_gemm_nt_nopad_contig_asm,"m_grouped_w8a8_gemm_nt_nopad_contig_asm");
  m.def("gemm_w8a8_smooth_asm", &at::native::gemm_w8a8_smooth_asm, "gemm_w8a8_smooth_asm",
          py::arg("A"), py::arg("B"), py::arg("As"), py::arg("Bs"), py::arg("bias"),
          py::arg("output_dtype"), py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("isTunning_slide_block"), py::arg("tunning_slide_block"),
          py::arg("input_scale_first") = false);
  m.def("moe_w16a16_asm", &at::native::moe_w16a16_asm, "moe_w16a16_asm");
  m.def("moe_marlin_w16a16_asm", &at::native::moe_marlin_w16a16_asm, "moe_marlin_w16a16_asm");
  m.def("gemm_w8a8_asm", &at::native::gemm_w8a8_asm, "gemm_w8a8_block_asm");
  m.def("moe_w8a8_asm", &at::native::moe_w8a8_asm, "moe_w8a8_asm");
  m.def("moe_marlin_w8a8_asm", &at::native::moe_marlin_w8a8_asm, "moe_marlin_w8a8_asm");
  m.def("moe_marlin_w8a8_fp8_asm", &at::native::moe_marlin_w8a8_fp8_asm, "moe_marlin_w8a8_fp8_asm");
  m.def("moe_w4a8_asm", &at::native::moe_w4a8_asm, "moe_w4a8_asm");
  m.def("moe_w4a8_marlin_asm", &at::native::moe_w4a8_marlin_asm, "moe_w4a8_marlin_asm");
  m.def("moe_groupgemm_w4a8_asm", &at::native::moe_groupgemm_w4a8_asm, "moe_groupgemm_w4a8_asm");
  m.def("moe_groupgemm_marlin_w4a8_asm", &at::native::moe_groupgemm_marlin_w4a8_asm, "moe_groupgemm_marlin_w4a8_asm");
  m.def("m_grouped_marlin_w4a8_gemm_nt_masked_asm", &at::native::m_grouped_marlin_w4a8_gemm_nt_masked_asm, "m_grouped_marlin_w4a8_gemm_nt_masked_asm");
  m.def("m_grouped_marlin_w8a8_gemm_nt_masked_asm", &at::native::m_grouped_marlin_w8a8_gemm_nt_masked_asm, "m_grouped_marlin_w8a8_gemm_nt_masked_asm");
  m.def("m_grouped_marlin_fp8_gemm_nt_masked_asm", &at::native::m_grouped_marlin_fp8_gemm_nt_masked_asm, "m_grouped_marlin_fp8_gemm_nt_masked_asm");
  m.def("m_grouped_bf16_gemm_nt_masked_asm", &at::native::m_grouped_bf16_gemm_nt_masked_asm, "m_grouped_bf16_gemm_nt_masked_asm");
  m.def("groupgemm",&at::native::groupgemm,"groupgemm");
}
