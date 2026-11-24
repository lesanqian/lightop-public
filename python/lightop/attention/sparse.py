# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sparse-attention index transformation APIs."""

from .. import op as lightop_native
from ..fuse_topk_transform import (
    fast_topk_transform_fused,
    fast_topk_transform_sparse_mask_fused,
    fast_topk_transform_ragged_fused,
    topk_transform_512,
    topk_transform_1024,
    topk_transform_ragged_v2,
    fast_kpool_topk_transform_fused,
)

# These native ops may be absent in a selective build. Resolve them on access
# so the remaining exports of this module stay usable.
_LAZY_OP_ALIASES = (
    "convert_req_index_to_global_index",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
)


def __getattr__(name):
    if name in _LAZY_OP_ALIASES:
        return getattr(lightop_native, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "topk_transform_1024",
    "topk_transform_ragged_v2",
    "fast_kpool_topk_transform_fused",
    "convert_req_index_to_global_index",
    "fast_topk_transform_fused",
    "fast_topk_transform_sparse_mask_fused",
    "fast_topk_transform_ragged_fused",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
    "topk_transform_512",
]
