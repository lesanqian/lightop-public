# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Content-hash module fingerprints for LightOp's incremental native build.

This module is imported by ``setup.py`` (through an explicit file loader, so it
works regardless of ``sys.path``) and is deliberately stdlib-only so it can also
be unit tested without torch/hipcc.

Design (see lightop_aiter_build_analysis.md §8):
  * Layer 1 skips a whole module when its fingerprint equals the cached value and
    the target ``.so`` already exists. The fingerprint is a SHA-256 over the
    module's compile signature plus the *contents* of every source/header it
    depends on. Contents (not mtimes) are hashed because a ``git checkout`` can
    roll mtimes back and silently hide an edit.
  * Layer 2 (per-module Ninja + ``hipcc -MMD``) handles the fine-grained rebuild
    once a module is known to be dirty.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess

# Bump when the fingerprint algorithm changes so every cache is invalidated.
_FINGERPRINT_VERSION = "lightop-incremental-v2"
_FINGERPRINT_DIRNAME = ".lightop_fingerprints"
_READ_CHUNK = 1 << 20


def sha256_file(path):
    """Return the hex SHA-256 of a file's full contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_make_words(text):
    """Split a Make dependency list, honouring backslash-escaped spaces."""
    words, current, index = [], [], 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] == " ":
            current.append(" ")
            index += 2
            continue
        if char.isspace():
            if current:
                words.append("".join(current))
                current = []
            index += 1
            continue
        current.append(char)
        index += 1
    if current:
        words.append("".join(current))
    return words


def parse_depfile(path):
    """Return the dependency paths listed in a Make-style ``.d`` file."""
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        content = stream.read()
    content = content.replace("\\\r\n", " ").replace("\\\n", " ")
    dependencies = []
    for line in content.splitlines():
        if ":" not in line:
            continue
        _, _, right = line.partition(":")
        dependencies.extend(_split_make_words(right))
    return dependencies


def collect_dependency_files(search_dirs, repo_root):
    """Repository-local sources/headers referenced by ``*.d`` under search_dirs.

    Only files inside ``repo_root`` are returned: toolchain headers (hiprand,
    rocrand, ...) are intentionally ignored because toolchain changes are
    captured by the build signature instead of by content hashes.
    """
    repo_root = os.path.realpath(repo_root)
    prefix = repo_root + os.sep
    found = set()
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for depfile in glob.glob(os.path.join(directory, "**", "*.d"), recursive=True):
            for dependency in parse_depfile(depfile):
                candidate = dependency if os.path.isabs(dependency) else os.path.join(directory, dependency)
                candidate = os.path.realpath(candidate)
                if candidate.startswith(prefix) and os.path.isfile(candidate):
                    found.add(candidate)
                    # The compiler records hipified *copies* (``foo_hip.h``);
                    # the developer edits the original (``foo.h``), so watch it too.
                    original = _original_from_hip(candidate)
                    if original and os.path.isfile(original) \
                            and os.path.realpath(original).startswith(prefix):
                        found.add(os.path.realpath(original))
    return found


def _original_from_hip(path):
    """Return the un-hipified source of a ``*_hip.<ext>`` file, or ``None``.

    torch hipify writes ``<name>_hip.<ext>`` next to the original when building
    for ROCm, and the compiler's dependency list then references only those
    generated copies. Watching the original is what lets an edit to the real
    source header invalidate the module before hipify runs again.
    """
    base = os.path.basename(path)
    match = re.search(r"_hip(\.[^.]+)$", base)
    if not match:
        return None
    return os.path.join(os.path.dirname(path), base[:match.start()] + match.group(1))


def parse_ninja_deps_output(text):
    """Return the dependency paths from ``ninja -t deps`` output.

    Each output record has an unindented header line such as
    ``build/.../foo.o: #deps 12, deps mtime 123 (VALID)``; the dependencies
    follow on indented lines.
    """
    dependencies = []
    for line in text.splitlines():
        if not line[:1].isspace():
            continue
        candidate = line.strip()
        if candidate and candidate != "#":
            dependencies.append(candidate)
    return dependencies


def collect_ninja_deps(module_build_dir, repo_root, timeout=120):
    """Repository-local headers recorded in a module's ``.ninja_deps``.

    Ninja stores ``deps = gcc`` dependency files inside the binary
    ``.ninja_deps`` database and *deletes* the ``*.o.d`` text files after
    reading them, so on the next build ``collect_dependency_files`` finds
    nothing. We let Ninja dump the database (``ninja -C <dir> -t deps``) instead
    of parsing its private binary format, then keep the files under
    ``repo_root`` together with their de-hipified originals (see
    ``_original_from_hip``).
    """
    if not module_build_dir or not os.path.isdir(module_build_dir):
        return set()
    if not os.path.exists(os.path.join(module_build_dir, ".ninja_deps")):
        return set()
    executable = shutil.which("ninja")
    if not executable:
        return set()
    try:
        completed = subprocess.run(
            [executable, "-C", module_build_dir, "-t", "deps"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    repo_root = os.path.realpath(repo_root)
    prefix = repo_root + os.sep
    found = set()
    for dependency in parse_ninja_deps_output(completed.stdout.decode("utf-8", "replace")):
        candidate = dependency if os.path.isabs(dependency) else os.path.join(module_build_dir, dependency)
        candidate = os.path.realpath(candidate)
        if candidate.startswith(prefix) and os.path.isfile(candidate):
            found.add(candidate)
            original = _original_from_hip(candidate)
            if original and os.path.isfile(original) and os.path.realpath(original).startswith(prefix):
                found.add(os.path.realpath(original))
    return found


def _hipify_variants(name):
    """Spellings a header may take before/after torch hipify.

    ``reduce.h`` becomes ``reduce_hip.h`` (``_hip`` inserted before the suffix)
    and ``cuda_meta.h`` becomes ``hip_meta.h`` (``cuda_`` -> ``hip_``), so a
    dependency name recorded by the compiler will not literally equal the name
    of the file the developer edits.
    """
    variants = {name}
    match = re.match(r"^(.*?)(\.[^.]+)$", name)
    if match:
        variants.add(match.group(1) + "_hip" + match.group(2))
    if name.startswith("cuda_"):
        variants.add("hip_" + name[5:])
    if name.startswith("hip_"):
        variants.add("cuda_" + name[4:])
    return variants


def ninja_dependency_names(module_build_dir, timeout=120):
    """Basenames of every file in a module's ``.ninja_deps`` database.

    Unlike :func:`collect_ninja_deps` this keeps names that do not resolve to an
    existing file (for example hipify-generated ``*_hip.h`` copies). setup.py
    uses it to include only the shared ``csrc/include`` headers a module really
    pulls in, instead of rebuilding every module whenever any shared header
    changes.
    """
    if not module_build_dir or not os.path.isdir(module_build_dir):
        return set()
    if not os.path.exists(os.path.join(module_build_dir, ".ninja_deps")):
        return set()
    executable = shutil.which("ninja")
    if not executable:
        return set()
    try:
        completed = subprocess.run(
            [executable, "-C", module_build_dir, "-t", "deps"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    names = set()
    for dependency in parse_ninja_deps_output(completed.stdout.decode("utf-8", "replace")):
        names.add(os.path.basename(dependency))
    return names


def shared_header_referenced(header_basename, dependency_names):
    """True if a shared header (any hipified spelling) is a module dependency."""
    return bool(_hipify_variants(header_basename) & set(dependency_names))


def compute_fingerprint(repo_root, files, build_signature):
    """Return ``(hexdigest, {relpath: sha256})`` over repository-local files."""
    repo_root = os.path.realpath(repo_root)
    prefix = repo_root + os.sep
    entries = {}
    for path in {os.path.realpath(item) for item in files}:
        if not os.path.isfile(path) or not path.startswith(prefix):
            continue
        entries[os.path.relpath(path, repo_root)] = sha256_file(path)
    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(build_signature, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    for relpath in sorted(entries):
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entries[relpath].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), entries


def _sanitize_key(key):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(key)) or "module"


def fingerprint_path(build_root, key):
    """Path of the JSON cache for one module, sanitising the key for a filename."""
    return os.path.join(build_root, _FINGERPRINT_DIRNAME, _sanitize_key(key) + ".json")


def load_fingerprint(path):
    """Load a cached fingerprint dict, or ``None`` if absent/corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_fingerprint(path, data):
    """Atomically write a fingerprint dict (temp file + ``os.replace``)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=1, sort_keys=True)
    os.replace(temporary, path)


def describe_changes(previous, entries, build_signature):
    """Human-readable explanation of why a cached fingerprint no longer matches."""
    if not previous:
        return "no cached fingerprint"
    reasons = []
    previous_build = previous.get("build")
    if previous_build is not None and previous_build != build_signature:
        reasons.append("toolchain/compile flags changed")
    previous_files = previous.get("files") or {}
    changed = [name for name in entries if previous_files.get(name) != entries[name]]
    removed = [name for name in previous_files if name not in entries]
    if changed:
        sample = ", ".join(sorted(changed)[:4])
        suffix = ", ..." if len(changed) > 4 else ""
        reasons.append(f"{len(changed)} source/header file(s) changed ({sample}{suffix})")
    if removed:
        reasons.append(f"{len(removed)} file(s) no longer present")
    return "; ".join(reasons) if reasons else "content changed"
