# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Direct bindings for native KV-cache operators."""

from .. import op as lightop_native

_LAZY_OP_ALIASES = (
    "fuse_act_quant_and_store_index_k_cache",
    "fuse_qk_quant_and_store_index_k_cache",
    "fused_quantize_and_store_mla_kv_cache",
    "quantize_nope_fp8_rope_bf16_pack_store",
    "fused_concat_and_store_mla_kv_cache",
)

__all__ = list(_LAZY_OP_ALIASES)

if hasattr(lightop_native, "fused_quantize_and_store_mla_kv_cache_dual"):
    fused_quantize_and_store_mla_kv_cache_dual = lightop_native.fused_quantize_and_store_mla_kv_cache_dual
    __all__.append("fused_quantize_and_store_mla_kv_cache_dual")

if hasattr(lightop_native, "main_kv_loc_to_compact"):
    main_kv_loc_to_compact = lightop_native.main_kv_loc_to_compact
    __all__.append("main_kv_loc_to_compact")

for _name in ('indexer_k_norm_rope_hadamard_store', 'fill_swa_ring_mapping'):
    if hasattr(lightop_native, _name):
        globals()[_name] = getattr(lightop_native, _name)
        __all__.append(_name)


def __getattr__(name):
    if name in _LAZY_OP_ALIASES:
        return getattr(lightop_native, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
