// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for the quantization, scaling, and FP8 packing family.

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/macros/Macros.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace at::native {
void blockwise_scaled_cast_transpose(const Tensor& input,std::optional<Tensor> output,std::optional<Tensor> scale_inv,
                                    std::optional<Tensor> output_t,std::optional<Tensor> scale_inv_t,std::optional<Tensor> noop,
                                    const int block_len,const int block_scaling_dim,const bool rowwise,const bool columnwise,
                                    const float epsilon,const bool pow2_scale,const bool all_gather_usage, at::ScalarType out_dtype);
void destindex_copy_quantize_int8_cuda(const torch::Tensor& input, torch::Tensor& scale_zero, torch::Tensor& output);
void destindex_dequantize_int8_cuda(const torch::Tensor& input, const torch::Tensor& scale_zero, torch::Tensor& output);
void per_token_dynamic_quant_int8(torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scales, torch::Tensor const& smoothscale, bool round_after_smooth, double limit);
void rope_per_token_quant_out(const torch::Tensor &, const torch::Tensor &,
                              const torch::Tensor &, torch::Tensor &,
                              torch::Tensor &, int64_t, int64_t, bool,
                              int64_t, bool);
void scale_per_token_quant_int8_out(
    const torch::Tensor &, const torch::Tensor &, const torch::Tensor &,
    std::optional<torch::Tensor>, torch::Tensor &, torch::Tensor &,
    at::ScalarType, bool, int64_t, int64_t);
void per_token_quant_int8_minimax_out(torch::Tensor&, const torch::Tensor&, torch::Tensor&, int64_t);
void per_token_quant_int8_out(torch::Tensor& out, const torch::Tensor& input, torch::Tensor& scales,
    const c10::optional<torch::Tensor>& smooth, const c10::optional<torch::Tensor>& weights,
    const c10::optional<torch::Tensor>& gate_output);
torch::Tensor fake_quantize_per_tensor_affine_custom(
    const torch::Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max);
torch::Tensor fake_quantize_per_channel_affine_custom(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max);
void per_token_quant_fp8(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales, int64_t quant_ndim, int64_t threads);
void per_token_group_quant_fp8(
    torch::Tensor& out,        
    torch::Tensor  input, 
    torch::Tensor& scales,
    int group_size,            
    float eps = 1e-5f,         
    bool use_ue8m0 = false);
void fuse_act_quant_and_store_index_k_cache(
    torch::Tensor input,      
    torch::Tensor buf,     
    torch::Tensor loc,    
    int page_size,
    float eps = 1e-5f,         
    bool use_ue8m0 = false,
    bool is_e4m3 = true);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fuse_qk_quant_and_store_index_k_cache(
    torch::Tensor q_input,        // [num_tokens, q_hidden_size]
    torch::Tensor k_input,        // [num_tokens, k_hidden_size]
    torch::Tensor k_buf,          // The destination Paged Cache Buffer (uint8)
    torch::Tensor k_loc,          // [num_tokens] the cache block indices (int64)
    int page_size,
    c10::optional<torch::Tensor> weights_in_opt = c10::nullopt,
    float q_scale_factor = 1.0f,  
    float k_scale_factor = 1.0f,  
    float eps = 1e-5f,        
    bool use_ue8m0 = false,
    bool is_e4m3 = true);
torch::Tensor minimax_cp_fp8_pack(torch::Tensor,torch::Tensor,torch::Tensor,double,double,bool,bool);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blockwise_scaled_cast_transpose", &at::native::blockwise_scaled_cast_transpose, "blockwise_scaled_cast_transpose");
  m.def("destindex_copy_quantize_int8_cuda", &at::native::destindex_copy_quantize_int8_cuda, "destindex_copy_quantize_int8_cuda");
  m.def("destindex_dequantize_int8_cuda", &at::native::destindex_dequantize_int8_cuda, "destindex_dequantize_int8_cuda");
  m.def(
        "per_token_dynamic_quant_int8",
        &at::native::per_token_dynamic_quant_int8,
        "per_token_dynamic_quant_int8",
        py::arg("out"),
        py::arg("input"),
        py::arg("scales"),
        py::arg("smoothscale"),
        py::arg("round_after_smooth") = false,
        py::arg("limit") = -1.0);
  m.def("rope_per_token_quant_out", &at::native::rope_per_token_quant_out);
  m.def("scale_per_token_quant_int8_out",
          &at::native::scale_per_token_quant_int8_out);
  m.def("per_token_quant_int8_minimax_out", &at::native::per_token_quant_int8_minimax_out);
  m.def(
        "per_token_quant_int8_out",
        &at::native::per_token_quant_int8_out,
        "Per-token symmetric INT8 quantization (out variant)",
        py::arg("out"),
        py::arg("input"),
        py::arg("scales"), py::arg("smooth") = py::none(),
        py::arg("weights") = py::none(), py::arg("gate_output") = py::none());
  m.def("fake_quantize_per_tensor_affine_custom", &at::native::fake_quantize_per_tensor_affine_custom, "fake_quantize_per_tensor_affine_custom");
  m.def("fake_quantize_per_channel_affine_custom", &at::native::fake_quantize_per_channel_affine_custom, "fake_quantize_per_channel_affine_custom");
  m.def("per_token_quant_fp8", &at::native::per_token_quant_fp8,
         py::arg("out"), py::arg("input"), py::arg("scales"),
         py::arg("quant_ndim") = 1, py::arg("threads") = 256);
  m.def("per_token_group_quant_fp8", &at::native::per_token_group_quant_fp8, "per_token_group_quant_fp8");
  m.def("fuse_act_quant_and_store_index_k_cache", &at::native::fuse_act_quant_and_store_index_k_cache, "fuse_act_quant_and_store_index_k_cache");
  m.def("fuse_qk_quant_and_store_index_k_cache", &at::native::fuse_qk_quant_and_store_index_k_cache, "fuse_qk_quant_and_store_index_k_cache");
  m.def("minimax_cp_fp8_pack", &at::native::minimax_cp_fp8_pack);
  m.attr("per_token_quant_int8_out_supports_gate") = true;
  m.attr("PER_TOKEN_QUANT_FP8_VERSION") = 2;
}
