# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native expert-parallel data movement APIs."""

from .. import op as lightop_native

_NATIVE_EXPORTS = (
    "ep_build_m_indices",
    "ep_gather",
    "ep_scatter",
)

__all__ = list(_NATIVE_EXPORTS)


def __getattr__(name):
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(lightop_native, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
