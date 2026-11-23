# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public quantization, dequantization, and format-conversion APIs."""

from importlib import import_module

from ..minimax_cp_fp8_pack import minimax_cp_fp8_pack
from .affine import w4tow16
from .blockwise import blockwise_scaled_cast_transpose
from .fp8 import (
    fp8_to_16,
    per_token_group_quant_fp8,
    per_token_quant_fp8,
)


def __getattr__(name):
    if name == "quantize_nope_fp8_rope_bf16_pack":
        value = getattr(import_module(".fp8", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
from .int8 import (
    destindex_copy_quantize_int8_cuda,
    destindex_dequantize_int8_cuda,
    per_token_dynamic_quant_int8,
    per_token_group_quant_int8,
    per_token_quant_int8,
)
from .transforms import rope_per_token_quant, scale_per_token_quant_int8

__all__ = [
    "rope_per_token_quant",
    "scale_per_token_quant_int8",
    "minimax_cp_fp8_pack",
    "blockwise_scaled_cast_transpose",
    "destindex_copy_quantize_int8_cuda",
    "destindex_dequantize_int8_cuda",
    "fp8_to_16",
    "per_token_group_quant_fp8",
    "per_token_quant_fp8",
    "quantize_nope_fp8_rope_bf16_pack",
    "per_token_dynamic_quant_int8",
    "per_token_group_quant_int8",
    "per_token_quant_int8",
    "w4tow16",
]
