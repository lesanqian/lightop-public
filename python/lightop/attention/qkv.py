# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Direct bindings for existing LightOp QKV APIs."""

from .. import op as lightop_native
from ..fuse_rmsnorm_rope import (
    gammarms_rotary_embedding_fuse_with_kv_store,
    gammarms_rotary_embedding_fuse_with_kv_store_fake,
    mimo_v2_split_rope_vscale_kv_store,
    rms_rotary_embedding_fuse,
    rms_rotary_embedding_fuse_with_kv_store,
    rms_rotary_embedding_fuse_with_kv_store_fake,
    split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant,
)
from ..rmsnorm import (
    fuse_rmsnorm_rope_quant_qkv,
    fused_rms_norm_rope_contiguous,
)


__all__ = [
    "fuse_layernorm_rotary_embedding",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
    "gemma_qkv_split_rms_rotary_embedding_fuse",
    "gammarms_rotary_embedding_fuse_with_kv_store",
    "gammarms_rotary_embedding_fuse_with_kv_store_fake",
    "mimo_v2_split_rope_vscale_kv_store",
    "rms_rotary_embedding_fuse",
    "rms_rotary_embedding_fuse_with_kv_store",
    "rms_rotary_embedding_fuse_with_kv_store_fake",
    "split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant",
    "fuse_rmsnorm_rope_quant_qkv",
    "fused_rms_norm_rope_contiguous",
]

for _name in ('spark_mla_rope_cat_store', 'indexer_q_rope_hadamard_quant', 'mla_q_absorb_rope_concat'):
    if hasattr(lightop_native, _name):
        globals()[_name] = getattr(lightop_native, _name)
        __all__.append(_name)


_NATIVE_EXPORTS = (
    "fuse_layernorm_rotary_embedding",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope",
    "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
    "gemma_qkv_split_rms_rotary_embedding_fuse",
)


def __getattr__(name):
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(lightop_native, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
