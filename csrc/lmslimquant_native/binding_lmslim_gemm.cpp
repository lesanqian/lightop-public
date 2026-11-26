// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for LMSlimQuant native and AWQ W4A16 GEMM implementations.

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "bindings.h"
#include "hipblaslt/blaslt_w8a8_gemm.hpp"
#include "hipblaslt/blaslt_w8a8_gemm_v2.hpp"

namespace at::native {
void gemm_awq_w4a16(Tensor& ret,const Tensor& x,const Tensor& y,const Tensor& scales_zeros,int num_n_warps, int num_k_warps, int n_per_warp,int k_per_warp,int splitk,bool use_marlin);
void gemm_awq_w4a16_dequant(Tensor& ret,const Tensor& weight,const Tensor& scales_zeros,bool use_marlin);
torch::Tensor awq_w4a16_dequant_trans(
    torch::Tensor _weight, 
    torch::Tensor _scaling_factors, 
    torch::Tensor _zeros, 
    int IC, 
    int OC, 
    int group_size);
}  // namespace at::native

namespace lightop_int8_blaslt {
std::optional<torch::Tensor> scaled_mm(
    const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& scale_input, const torch::Tensor& scale_weight,
    int solution);
}  // namespace lightop_int8_blaslt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_lmslimquant_rocblas_gemm_int8",
          &lightop::lmslimquant_native::rocblas_gemm_int8);
  m.def("_lmslimquant_rocblas_gemm_fp16",
          &lightop::lmslimquant_native::rocblas_gemm_fp16);
  m.def("_lmslimquant_gemm_w4a16_v2",
          &lightop::lmslimquant_native::gemm_w4a16_v2);
  m.def("_lmslimquant_awq_input_padding",
          &lightop::lmslimquant_native::awq_input_padding);
  m.def("_lmslimquant_convert_s4_k_m8",
          &lightop::lmslimquant_native::convert_s4_k_m8);
  m.def("_lmslimquant_permute_trans",
          &lightop::lmslimquant_native::permute_trans);
  m.def("_lmslimquant_awq_dequant_colmajor",
          &lightop::lmslimquant_native::awq_dequant_colmajor);
  m.def("_lmslimquant_dequant_w4_gemm_colmajor",
          &lightop::lmslimquant_native::dequant_w4_gemm_colmajor);
  m.def("_lmslimquant_gptq_gemm",
          &lightop::lmslimquant_native::gptq_gemm_native);
  m.def("_lmslimquant_gptq_shuffle",
          &lightop::lmslimquant_native::gptq_shuffle_native);
  m.def("_lmslimquant_hipblaslt_gemm_channelwise",
          &hipblaslt_gemm_v2::hipblaslt_gemm_channelwise);
  m.def("_lmslimquant_hipblaslt_gemm_channelwise_kme",
          &hipblaslt_gemm_v2::hipblaslt_gemm_channelwise_kme);
  m.def("_lmslimquant_hipblaslt_gemm_blockwise",
          &hipblaslt_gemm_v2::hipblaslt_gemm_blockwise);
  m.def("_lmslimquant_blaslt_w8a8_fp16_gemm",
          &hipblaslt_gemm::w8a8_fp16_gemm,
          py::arg("mat_a"), py::arg("mat_b"), py::arg("scale_a"),
          py::arg("scale_b"), py::arg("output"), py::arg("m"), py::arg("n"),
          py::arg("k"), py::arg("batch"), py::arg("transpose_flag"),
          py::arg("alpha"), py::arg("beta"), py::arg("bias") = py::none());
  m.def("_lmslimquant_blaslt_w8a8_bf16_gemm",
          &hipblaslt_gemm::w8a8_bf16_gemm,
          py::arg("mat_a"), py::arg("mat_b"), py::arg("scale_a"),
          py::arg("scale_b"), py::arg("output"), py::arg("m"), py::arg("n"),
          py::arg("k"), py::arg("batch"), py::arg("transpose_flag"),
          py::arg("alpha"), py::arg("beta"), py::arg("bias") = py::none());
  m.def("_lmslimquant_blaslt_w8a8_bf16_gemm_cached",
          &lightop_int8_blaslt::scaled_mm,
          py::arg("input"), py::arg("weight"), py::arg("scale_input"),
          py::arg("scale_weight"), py::arg("solution"));
  m.def("gemm_awq_w4a16", &at::native::gemm_awq_w4a16, "gemm_awq_w4a16");
  m.def("gemm_awq_w4a16_dequant", &at::native::gemm_awq_w4a16_dequant, "gemm_awq_w4a16_dequant");
  m.def("awq_w4a16_dequant_trans",
            &at::native::awq_w4a16_dequant_trans, 
            py::arg("_weight"), 
            py::arg("_scaling_factors"), 
            py::arg("_zeros"), 
            py::arg("IC"), 
            py::arg("OC"), 
            py::arg("group_size"));
}
