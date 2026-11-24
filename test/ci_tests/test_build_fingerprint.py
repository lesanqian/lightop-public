# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the incremental-build fingerprint helper (no torch/hipcc needed).

These exercise build_fingerprint.py, the layer-1 module-skip primitive used by
setup.py: content hashing, depfile parsing and change reporting.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "lightop_build_fingerprint", str(_REPO_ROOT / "build_fingerprint.py"))
fp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fp)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_sha256_file_is_content_based(tmp_path):
    first = tmp_path / "a.cu"
    _write(first, "kernel v1\n")
    digest = fp.sha256_file(str(first))
    assert digest == fp.sha256_file(str(first))
    _write(first, "kernel v2\n")
    assert fp.sha256_file(str(first)) != digest


def test_parse_depfile_handles_continuations_and_escaped_spaces(tmp_path):
    depfile = tmp_path / "obj.o.d"
    depfile.write_text(
        "obj.o: /repo/a.cu /repo/with\\ space.h \\\n"
        "  /repo/b.h\n",
        encoding="utf-8")
    assert fp.parse_depfile(str(depfile)) == [
        "/repo/a.cu", "/repo/with space.h", "/repo/b.h"]


def test_collect_dependency_files_filters_outside_repo(tmp_path):
    repo = tmp_path / "repo"
    build = repo / "build" / "objdir"
    build.mkdir(parents=True)
    _write(repo / "src" / "a.cu", "a")
    _write(repo / "include" / "common.h", "h")
    _write(build / "a.o.d",
           f"a.o: {repo / 'src' / 'a.cu'} {repo / 'include' / 'common.h'} /opt/rocm/x.h\n")
    found = fp.collect_dependency_files([str(build)], str(repo))
    rel = {Path(os.path.relpath(path, str(repo))).as_posix() for path in found}
    assert rel == {"src/a.cu", "include/common.h"}


def test_compute_fingerprint_detects_source_and_header_edits(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "src" / "a.cu"
    header = repo / "include" / "common.h"
    _write(source, "kernel\n")
    _write(header, "header v1\n")
    watch = {str(source), str(header)}
    signature = {"nvcc": ["-O3"], "dtk": "2604"}

    base, entries = fp.compute_fingerprint(str(repo), watch, signature)
    again, _ = fp.compute_fingerprint(str(repo), watch, signature)
    assert base == again
    assert {Path(name).as_posix() for name in entries} == {"src/a.cu", "include/common.h"}

    _write(header, "header v2\n")
    after_header, _ = fp.compute_fingerprint(str(repo), watch, signature)
    assert after_header != base

    _write(source, "kernel changed\n")
    after_source, _ = fp.compute_fingerprint(str(repo), watch, signature)
    assert after_source not in (base, after_header)


def test_compute_fingerprint_detects_signature_change(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "a.cu"
    _write(source, "kernel\n")
    watch = {str(source)}
    one, _ = fp.compute_fingerprint(str(repo), watch, {"arch": "gfx928"})
    two, _ = fp.compute_fingerprint(str(repo), watch, {"arch": "gfx938"})
    assert one != two


def test_fingerprint_path_sanitizes_key(tmp_path):
    path = fp.fingerprint_path(str(tmp_path), "module:mqa_logits")
    assert path.endswith(os.path.join(".lightop_fingerprints", "module_mqa_logits.json"))


def test_write_load_roundtrip_and_describe_changes(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "a.cu"
    _write(source, "kernel\n")
    signature = {"nvcc": ["-O3"]}
    digest, entries = fp.compute_fingerprint(str(repo), {str(source)}, signature)
    cache = fp.fingerprint_path(str(tmp_path / "build"), "module:rmsnorm")
    fp.write_fingerprint(cache, {"fingerprint": digest, "build": signature, "files": entries})
    assert fp.load_fingerprint(cache)["fingerprint"] == digest

    _write(source, "kernel edited\n")
    _, new_entries = fp.compute_fingerprint(str(repo), {str(source)}, signature)
    reason = fp.describe_changes(fp.load_fingerprint(cache), new_entries, signature)
    assert "a.cu" in reason


def test_parse_ninja_deps_output_extracts_indented_paths():
    text = (
        "build/obj.o: #deps 3, deps mtime 12345 (VALID)\n"
        "    /repo/src/a.cu\n"
        "    /repo/include/common.h\n"
        "\n"
        "build/other.o: #deps 1, deps mtime 999 (VALID)\n"
        "    /repo/include/only.h\n"
    )
    assert fp.parse_ninja_deps_output(text) == [
        "/repo/src/a.cu", "/repo/include/common.h", "/repo/include/only.h"]


def test_original_from_hip_maps_generated_to_source(tmp_path):
    directory = tmp_path / "csrc" / "include"
    assert fp._original_from_hip(str(directory / "reduce_hip.h")) == str(directory / "reduce.h")
    assert fp._original_from_hip(str(directory / "m16_hip.hpp")) == str(directory / "m16.hpp")
    # "..._hip_common.h" is a real name, not a hipified copy.
    assert fp._original_from_hip(str(directory / "aiter_hip_common.h")) is None


def test_collect_ninja_deps_missing_dir_returns_empty(tmp_path):
    assert fp.collect_ninja_deps(str(tmp_path / "does-not-exist"), str(tmp_path)) == set()
