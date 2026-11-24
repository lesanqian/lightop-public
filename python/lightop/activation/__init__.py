# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Activation, gating, dropout and drop-path APIs."""

from ..fuse_silu_mul_quant import (
    fuse_silu_mul_clamp_quant,
    fuse_silu_mul_clamp_quant_ep,
)
from ..swiglu_oai_fp8_quant_contiguous import swiglu_oai_fp8_quant_contiguous

from importlib import import_module

from .existing import (
    AddDropout,
    DropPath,
    FuseBiasDropoutAdd,
    FuseBiasGelu,
    FuseBiasSwiGLU,
    GNSilu,
    LnAddDropout,
    LnAddDropoutBias,
    LnDropPath,
    LnDropout,
    ReluDropout,
    fuse_silu_and_mul,
    fuse_silu_and_mul_ep,
    fuse_silu_mul_fp8_quant,
    fuse_silu_mul_fp8_quant_ep,
    fuse_silu_mul_per_token_quant,
    fuse_silu_mul_quant,
    fuse_silu_mul_quant_ep,
    fuse_situ_mul_quant_contiguous,
    fuse_situ_mul_quant_ep,
    relu2,
)

from .swiglu_oai_quant import swiglu_oai_fp8_quant_masked

_LAZY_EXPORTS = {
    name: module
    for module, names in {
        ".existing": (
            "gelu_and_mul_opt",
            "gelu_tanh_and_mul_opt",
            "silu_and_mul_opt",
        ),
        ".native": (
            "lm_fuse_silu_mul_quant",
            "fused_clamp_smooth",
        ),
    }.items()
    for name in names
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "fused_clamp_smooth",
    "swiglu_oai_fp8_quant_contiguous",

    "AddDropout",
    "DropPath",
    "fuse_silu_and_mul",
    "fuse_silu_and_mul_ep",
    "fuse_silu_mul_clamp_quant",
    "fuse_silu_mul_clamp_quant_ep",
    "fuse_silu_mul_fp8_quant",
    "fuse_silu_mul_fp8_quant_ep",
    "fuse_silu_mul_per_token_quant",
    "fuse_silu_mul_quant",
    "fuse_silu_mul_quant_ep",
    "fuse_situ_mul_quant_contiguous",
    "fuse_situ_mul_quant_ep",
    "FuseBiasDropoutAdd",
    "FuseBiasGelu",
    "FuseBiasSwiGLU",
    "GNSilu",
    "gelu_and_mul_opt",
    "gelu_tanh_and_mul_opt",
    "lm_fuse_silu_mul_quant",
    "LnAddDropout",
    "LnAddDropoutBias",
    "LnDropout",
    "LnDropPath",
    "relu2",
    "ReluDropout",
    "silu_and_mul_opt",
    "swiglu_oai_fp8_quant_masked",
]

from .swiglu_oai_int8_quant import (
    swiglu_oai_int8_quant_contiguous,
    swiglu_oai_int8_quant_masked,
    swiglu_oai_int8_quant_masked_flat,
)

__all__ += [
    "swiglu_oai_int8_quant_contiguous",
    "swiglu_oai_int8_quant_masked",
    "swiglu_oai_int8_quant_masked_flat",
]
