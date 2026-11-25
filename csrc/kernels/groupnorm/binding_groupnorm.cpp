// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for the GroupNorm family implemented by csrc/kernels/groupnorm/*.cu.

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace at::native {

Tensor fuse_groupnorm_forward_autograd(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& bias,
    int num_groups,
    int num_channels,
    double eps,
    bool training);

Tensor fuse_celu_gn_forward_autograd(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& bias,
    int num_groups,
    int num_channels,
    double eps,
    double alpha,
    bool training);

Tensor fuse_gn_silu_forward_autograd(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& bias,
    int num_groups,
    int num_channels,
    double eps,
    bool training);

}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("groupnorm_forward_autograd", &at::native::fuse_groupnorm_forward_autograd, "groupnorm forward");
  m.def("celu_gn_forward_autograd", &at::native::fuse_celu_gn_forward_autograd, "celu_gn forward");
  m.def("gn_silu_forward_autograd", &at::native::fuse_gn_silu_forward_autograd, "gn_silu forward");
}
