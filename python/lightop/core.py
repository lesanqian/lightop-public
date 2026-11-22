# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build metadata and explicit native-module loading. No JIT compilation.

Reading this file (including through runpy) only loads JSON. Native extensions
are imported exclusively by load_ops(), called during package initialization.
"""
import json
import importlib
import sys
from types import ModuleType
from pathlib import Path


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate registry key: {key!r}")
        result[key] = value
    return result


def load_registry(path=None):
    """Read JSON, validate build fields and supply optional defaults."""
    path = Path(path) if path is not None else Path(__file__).with_name("module_registry.json")
    with path.open(encoding="utf-8") as stream:
        modules = json.load(stream, object_pairs_hook=_unique_keys)
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Module registry must be a non-empty object")
    required = {"extension", "sources"}
    allowed = required | {"include_dirs", "extra_compile_args"}
    extensions = set()
    for name, spec in modules.items():
        if not name.isidentifier() or not isinstance(spec, dict):
            raise ValueError(f"Invalid module definition: {name!r}")
        if required - set(spec) or set(spec) - allowed:
            raise ValueError(f"{name}: missing fields {sorted(required - set(spec))}; "
                             f"unknown fields {sorted(set(spec) - allowed)}")
        extension = spec["extension"]
        if not isinstance(extension, str) or not extension:
            raise ValueError(f"{name}.extension: expected a non-empty string")
        if extension in extensions:
            raise ValueError(f"{name}: duplicate extension name")
        extensions.add(extension)
        spec.setdefault("include_dirs", [])
        spec.setdefault("extra_compile_args", {})
        for field in ("sources", "include_dirs"):
            values = spec[field]
            if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
                raise ValueError(f"{name}.{field}: expected a list of non-empty strings")
        if not spec["sources"]:
            raise ValueError(f"{name}.sources: at least one source is required")
        flags = spec["extra_compile_args"]
        if not isinstance(flags, dict) or any(
            key not in ("cxx", "nvcc") or not isinstance(values, list)
            or not all(isinstance(v, str) and v for v in values)
            for key, values in flags.items()
        ):
            raise ValueError(f"{name}.extra_compile_args: expected cxx/nvcc lists of strings")
    return modules


MODULES = load_registry()


def select_modules(environ):
    """Select modules only through LIGHTOP_MODULES, in registry order."""
    selected = {name.strip() for name in environ.get("LIGHTOP_MODULES", "").split(",")
                if name.strip()}
    unknown = sorted(selected - set(MODULES))
    if unknown:
        raise ValueError(f"LIGHTOP_MODULES: unknown module(s) {unknown}; "
                         f"known modules: {list(MODULES)}")
    return tuple(name for name in MODULES if name in selected)


class _SplitOps(ModuleType):
    """Stable public facade with exactly one native owner per exported symbol."""

    def __init__(self, backends):
        super().__init__("lightop.op")
        self.__file__ = __file__
        self._deleted_exports = set()
        self._owners = {}
        for backend in backends:
            for name in dir(backend):
                if name.startswith("__"):
                    continue
                if name in self._owners:
                    previous = self._owners[name].__name__
                    raise ImportError(
                        f"Duplicate LightOp native export {name!r}: "
                        f"{previous} and {backend.__name__}. Rebuild the extensions."
                    )
                self._owners[name] = backend
        self.__all__ = sorted(name for name in self._owners if not name.startswith("_"))

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        deleted = self.__dict__.get("_deleted_exports")
        if deleted is not None:
            deleted.discard(name)

    def __delattr__(self, name):
        # Resolve uncached exports before deleting, without mutating their owner.
        getattr(self, name)
        super().__delattr__(name)
        self._deleted_exports.add(name)

    def __getattr__(self, name):
        if name in self._deleted_exports:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        backend = self._owners.get(name)
        if backend is None:
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {name!r}; "
                "the required native module may not have been built"
            )
        value = getattr(backend, name)
        setattr(self, name, value)
        return value

    def __dir__(self):
        return sorted((set(super().__dir__()) | set(self._owners)) - self._deleted_exports)


def load_ops():
    """Combine registered family extensions and the optional remaining operators.

    Legacy op.so is never loaded: lightop.op always names the Python facade.
    Only an absent target extension is optional; dependency/ABI errors propagate.
    """
    existing = sys.modules.get("lightop.op")
    if isinstance(existing, _SplitOps):
        return existing
    backends = []
    names = [spec["extension"] for spec in MODULES.values()] + ["lightop._C"]
    for name in names:
        try:
            backends.append(importlib.import_module(name))
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    if not backends:
        raise ImportError(
            "No LightOp native extension found. Build the full package or select "
            "modules with LIGHTOP_MODULES=" + ",".join(MODULES) + "."
        )
    op = _SplitOps(backends)
    sys.modules["lightop.op"] = op
    return op
