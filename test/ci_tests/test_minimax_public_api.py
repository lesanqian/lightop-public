# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Installed-package API contracts; no source-tree-relative imports or GPU work."""

import importlib
from types import ModuleType

import pytest
import lightop


@pytest.mark.parametrize(
    "category,name",
    [
        ("attention", "minimax_qk_norm_rope"),
        ("attention", "supports_minimax_qk_norm_rope"),
        ("kvcache", "minimax_cp_cache_scatter"),
        ("quant", "minimax_cp_fp8_pack"),
        ("activation", "swiglu_oai_fp8_quant_contiguous"),
    ],
)
def test_classified_api_keeps_top_level_alias(category, name):
    module = importlib.import_module("lightop." + category)
    assert name in module.__all__
    assert getattr(module, name) is getattr(lightop, name)


@pytest.mark.parametrize(
    "name,args",
    [
        ("minimax_cp_fp8_pack", (None,) * 3),
        ("minimax_cp_cache_scatter", (None,) * 6),
        ("swiglu_oai_fp8_quant_contiguous", (None,)),
        ("minimax_qk_norm_rope", (None,) * 10 + (1e-5, 128, 64, True)),
    ],
)
def test_missing_native_binding_has_actionable_error(monkeypatch, name, args):
    module = importlib.import_module("lightop." + name)
    monkeypatch.delattr(lightop.op, name, raising=False)
    if name == "minimax_qk_norm_rope":
        assert not module.supports_minimax_qk_norm_rope()
    with pytest.raises(RuntimeError, match="LightOp.*(missing|not built)"):
        getattr(module, name)(*args)


@pytest.mark.parametrize("cached", [False, True])
def test_native_facade_delete_and_restore(cached):
    from lightop.core import _SplitOps

    backend = ModuleType("native_for_test")
    original = object()
    backend.operator = original
    facade = _SplitOps([backend])
    if cached:
        assert facade.operator is original

    del facade.operator
    assert not hasattr(facade, "operator")
    assert "operator" not in dir(facade)
    assert backend.operator is original
    with pytest.raises(AttributeError):
        del facade.operator
    with pytest.raises(AttributeError):
        del facade.unknown_operator

    facade.operator = original
    assert facade.operator is original
    assert "operator" in dir(facade)
    del facade.operator
    assert not hasattr(facade, "operator")
