// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Public interfaces implemented by csrc/kernels/layernorm/*.cu and
// csrc/kernels/groupnorm/*.cu.
// Keep signatures and Python defaults aligned with csrc/pybind/export.cpp.
#include <torch/extension.h>

#include <optional>
#include <string>
#include <vector>

namespace at::native {

Tensor fuse_layernorm_forward_autograd(const Tensor& x, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, double eps, bool training);
Tensor fuse_layernorm_mixed_fp32_params_autograd(const Tensor& x, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, double eps, bool training);
Tensor fuse_add_ln_forward_autograd(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& bias, double eps, bool training);
Tensor fuse_ln_dropout_forward_autograd(const Tensor& x, const Tensor& weight, const Tensor& bias, double eps, double rate, bool training);
Tensor fuse_ln_add_dropout_forward_autograd(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& bias, double eps, double rate, bool training);
Tensor fuse_ln_add_dropout_bias_forward_autograd(const Tensor& x, const Tensor& y, const Tensor& attention_bias, const Tensor& weight, const Tensor& bias, double eps, double rate, bool training);
Tensor fuse_layernorm_droppath_forward_autograd(const Tensor& x, const Tensor& weight, const Tensor& bias, double eps, double keep_prob, bool scale_by_keep, bool training);

Tensor fuse_groupnorm_forward_autograd(const Tensor& x, const Tensor& weight, const Tensor& bias, int num_groups, int num_channels, double eps, bool training);
Tensor fuse_celu_gn_forward_autograd(const Tensor& x, const Tensor& weight, const Tensor& bias, int num_groups, int num_channels, double eps, double alpha, bool training);
Tensor fuse_gn_silu_forward_autograd(const Tensor& x, const Tensor& weight, const Tensor& bias, int num_groups, int num_channels, double eps, bool training);

void layer_norm_opt(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, double epsilon, double limit);
void fused_add_layer_norm_opt(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& bias, double epsilon, double limit);

void fuse_layernorm_rotary_embedding(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox,
    std::optional<at::Tensor> weight_q,
    std::optional<at::Tensor> bias_q,
    std::optional<at::Tensor> weight_k,
    std::optional<at::Tensor> bias_k,
    std::optional<at::Tensor> residual_q,
    std::optional<at::Tensor> residual_k,
    double epsilon);

std::vector<at::Tensor> fuse_norm_rope_quant_fp8(
    at::Tensor positions,
    at::Tensor query,
    at::Tensor key,
    int64_t head_size,
    at::Tensor cos_sin_cache,
    bool is_neox,
    bool is_rmsnorm,
    std::optional<at::Tensor> weight_k,
    std::optional<at::Tensor> bias_k,
    double epsilon);

void layer_norm_fwd_1pass_opt(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    std::optional<torch::Tensor> B,
    std::optional<torch::Tensor> Z,
    std::optional<torch::Tensor> Mean,
    torch::Tensor& Rstd,
    int64_t stride_x_row,
    int64_t stride_y_row,
    int64_t stride_z_row,
    int64_t M,
    int64_t N,
    double eps,
    int64_t BLOCK_N,
    int64_t ROWS_PER_BLOCK,
    bool HAS_BIAS,
    bool HAS_Z,
    bool NORM_BEFORE_GATE,
    bool IS_RMS_NORM,
    const std::string& ACTIVATION);

}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp LayerNorm and GroupNorm operators";

  m.def("layernorm_forward_autograd", &at::native::fuse_layernorm_forward_autograd, "layernorm forward");
  m.def("layernorm_mixed_fp32_params_forward_autograd", &at::native::fuse_layernorm_mixed_fp32_params_autograd,
        "layernorm forward, activations half/bf16 and weight/bias float32");
  m.def("add_ln_forward_autograd", &at::native::fuse_add_ln_forward_autograd, "layernorm forward");
  m.def("ln_dropout_forward_autograd", &at::native::fuse_ln_dropout_forward_autograd, "fused_ln_dropout forward");
  m.def("ln_add_dropout_forward_autograd", &at::native::fuse_ln_add_dropout_forward_autograd, "fused_ln_add_dropout forward");
  m.def("ln_add_dropout_bias_forward_autograd", &at::native::fuse_ln_add_dropout_bias_forward_autograd, "fused_ln_add_dropout_bias forward");
  m.def("layernorm_droppath_forward_autograd", &at::native::fuse_layernorm_droppath_forward_autograd, "layernorm then droppath forward");

  m.def("groupnorm_forward_autograd", &at::native::fuse_groupnorm_forward_autograd, "groupnorm forward");
  m.def("celu_gn_forward_autograd", &at::native::fuse_celu_gn_forward_autograd, "celu_gn forward");
  m.def("gn_silu_forward_autograd", &at::native::fuse_gn_silu_forward_autograd, "gn_silu forward");

  m.def(
      "layer_norm_opt", &at::native::layer_norm_opt, "layer_norm_opt",
      py::arg("out"), py::arg("input"), py::arg("weight"), py::arg("bias"),
      py::arg("epsilon"), py::arg("limit") = -1.0);
  m.def(
      "fused_add_layer_norm_opt", &at::native::fused_add_layer_norm_opt,
      "fused_add_layer_norm_opt", py::arg("input"), py::arg("residual"),
      py::arg("weight"), py::arg("bias"), py::arg("epsilon"),
      py::arg("limit") = -1.0);

  m.def("fuse_layernorm_rotary_embedding", &at::native::fuse_layernorm_rotary_embedding, "fuse_layernorm_rotary_embedding");
  m.def("fuse_norm_rope_quant_fp8", &at::native::fuse_norm_rope_quant_fp8, "fuse_norm_rope_quant_fp8");

  m.def("layer_norm_fwd_1pass_opt", &at::native::layer_norm_fwd_1pass_opt,
        "layer_norm_fwd_1pass_opt",
        py::arg("X"),
        py::arg("Y"),
        py::arg("W"),
        py::arg("B"),
        py::arg("Z"),
        py::arg("Mean"),
        py::arg("Rstd"),
        py::arg("stride_x_row"),
        py::arg("stride_y_row"),
        py::arg("stride_z_row"),
        py::arg("M"),
        py::arg("N"),
        py::arg("eps"),
        py::arg("BLOCK_N"),
        py::arg("ROWS_PER_BLOCK"),
        py::arg("HAS_BIAS"),
        py::arg("HAS_Z"),
        py::arg("NORM_BEFORE_GATE"),
        py::arg("IS_RMS_NORM"),
        py::arg("ACTIVATION"));
}
