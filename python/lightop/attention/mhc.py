# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 mHC attention APIs."""

try:
    from .. import _mhc_repeat_cp as lightop_native
except ImportError:
    from .. import op as lightop_native

# These native ops may be absent in a selective build. Resolve them on access
# so importing this module never fails for an unbuilt family.
_LAZY_OP_ALIASES = (
    "mhc_repeat_cp_sglang",
    "supports_mhc_repeat_cp_sglang",
)

__all__ = list(_LAZY_OP_ALIASES)

# The existing public API stays usable with older full-library wheels.
if hasattr(lightop_native, "mhc_repeat_cp_sglang_out"):
    mhc_repeat_cp_sglang_out = lightop_native.mhc_repeat_cp_sglang_out
    __all__.append("mhc_repeat_cp_sglang_out")


def __getattr__(name):
    if name in _LAZY_OP_ALIASES:
        return getattr(lightop_native, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
