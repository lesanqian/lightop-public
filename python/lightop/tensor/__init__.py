# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""General tensor transformation and reduction APIs."""

from ..batch_linear_assignment import batch_linear_assignment
from ..ds_cat import ds_cat
from ..fill_zeros import fill_zeros
from ..reduce import mean, sum
from .. import op as lightop_native

_LAZY_OP_ALIASES = ("reshape_to_float",)


def __getattr__(name):
    if name in _LAZY_OP_ALIASES:
        return getattr(lightop_native, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "batch_linear_assignment",
    "ds_cat",
    "fill_zeros",
    "mean",
    "reshape_to_float",
    "sum",
]
