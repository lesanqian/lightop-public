#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unified gfx946 verification, ISA audit & benchmark suite for Gated RMSNorm / LayerNorm.

Reuses Group 1's 3-stage verification harness for `lightop.norm.layer_norm_fwd_1pass_opt`
(P0 -> C1 -> C2 -> C3, zero API change, SGLang `SGLANG_USE_LAYER_NORM_FWD` & Mamba2
strided `gate` layout contract):
  1. Stage 1 — Environment & Binary ISA Audit (offline + online):
     - `hy-smi` / `rocm-smi` / `rocminfo` / `torch` device detection.
     - `readelf --dyn-syms --wide` check (zero unresolved `__extendhfsf2`).
     - `hipprof --codeobj-analyze` excerpt when available.
     - `dccobjdump` disassembly audit on `gfx946` code objects in `op*.so`
       (also supports offline saved disassembly via `--isa-dump`):
       * Asserts `SCRATCH == 0` on all audited gfx946 `layer_norm_fwd_1pass_*` kernels.
       * Asserts distinct VGPR operands on `v_permlane{16,32}_swap_b32` (errata R4).
       * Asserts zero `v_cvt_pkrtz_` (RTZ forbidden; must use RNE `v_cvt_pk_{bf16,f16}_f32`).
       * Asserts zero IEEE FP32 division (`v_div_scale_f32`/`v_div_fmas_f32`) in
         gfx946 BF16/FP16 vectorized kernels (sigmoid uses 1-insn `v_rcp_f32`).
       * Asserts zero `v_dot2_f32_bf16` / `v_dot2_f32_f16` in Mamba2 pre-gate
         (`HAS_Z && !NORM_BEFORE_GATE && ACT != kNone`) and LayerNorm (`!IS_RMS_NORM`)
         kernels (must preserve strict FP32 `x * gate(z)` / centered variance).
       * Asserts zero FP64 VALU inside BF16/FP16 kernel sections.
       * Asserts zero `ds_bpermute_b32` in gfx946 BF16/FP16 vectorized kernels
         (pure-VALU DPP + `v_permlane{16,32}_swap_b32` reductions).
       * Tallies expected gfx946 instructions (`v_dot2_f32_bf16`, `v_dot2_f32_f16`,
         `v_pk_{add,mul,fma}_f32`, `v_cvt_pk_{bf16,f16}_f32`, `v_permlane{16,32}_swap_b32`,
         `v_mov_b32_dpp`, `v_rcp_f32`, `v_exp_f32`) and peak VGPR count per family.
  2. Stage 2 — Full Boundary Accuracy, Mamba2 Strided Gate & 4-XCD Consistency Stress:
     - Direct binding parity (`lightop.norm.layer_norm_fwd_1pass_opt is lightop.op.layer_norm_fwd_1pass_opt`)
       and invalid-input rejection checks (storage overflow, non-contiguous last dim).
     - Full 5-family dispatch matrix (`small_head_vec`, `small_head`, `vec`, `vec_multi`,
       `scalar`) x BF16/FP16/FP32 x RMSNorm/LayerNorm x bias x gate order x activation.
     - Mamba2 strided `gate` views (`pad in {1, 2, 4, 12, 16}`, `z_offset in {0, 7, 16}`,
       exact end-of-buffer `torch.split`, strided `X`/`Y` padding preservation).
     - Hardware edges: extreme `z` (`+-100`), LayerNorm large-mean cancellation (`mean=1000`),
       all-zero rows, `-0.0`, in-place `Y is X`, unaligned `offset=1`, `M=0`, `M=288`,
       HIP Graph capture/replay, large-batch `int64` stress, and 4-XCD determinism stress.
  3. Stage 3 — Decode / Prefill / Skewed Workload Benchmark + CSV Export:
     - Calibrates D2D HBM copy ceiling (`%CopyBW`) and compares LightOp vs SGLang Triton
       (auto-detected via `_try_build_triton_ln1p`, gracefully skipped if unavailable or
       `--skip-triton`) vs PyTorch eager FP32 baseline across Decode, Prefill, Mamba2
       Strided Gate, Large-N, LayerNorm, and Skewed/Imbalanced workloads.

Usage (from lightop repo root):
  python3 test/test_layer_norm_fwd_1pass_gfx946.py
  python3 test/test_layer_norm_fwd_1pass_gfx946.py --isa-only --strict-isa --so-path path/to/op.so
  python3 test/test_layer_norm_fwd_1pass_gfx946.py --isa-only --strict-isa --isa-dump path/to/dump.txt
  python3 test/test_layer_norm_fwd_1pass_gfx946.py --perfmodel --require-gfx946 --strict-isa
  python3 test/test_layer_norm_fwd_1pass_gfx946.py --require-gfx946 --strict-isa --csv gated_rmsnorm_946.csv
  pytest -v test/test_layer_norm_fwd_1pass_gfx946.py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Optional

torch: Any = None


def _ensure_torch(optional: bool = False) -> bool:
    global torch
    if torch is not None:
        return True
    try:
        import torch as _torch

        torch = _torch
        return True
    except ImportError:
        if optional:
            return False
        raise


def run_cmd(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"command timed out after {timeout}s: {' '.join(cmd)}"


def maybe_add_local_build_to_syspath() -> None:
    """Prefer repo-local `build/lib.*`, `--inplace` `python/lightop`, or sibling `lightop/build`."""
    if "lightop" in sys.modules:
        return

    here = Path(__file__).resolve()
    abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates: list[Path] = []
    for parent in here.parents:
        repo_build = parent / "build"
        if repo_build.is_dir():
            cands = sorted(repo_build.glob(f"lib.*{abi}")) or sorted(repo_build.glob("lib.*"))
            candidates.extend(reversed(cands))
        repo_python = parent / "python"
        if repo_python.is_dir():
            candidates.append(repo_python)
        sibling_build = parent / "lightop" / "build"
        if sibling_build.is_dir():
            cands = sorted(sibling_build.glob(f"lib.*{abi}")) or sorted(sibling_build.glob("lib.*"))
            candidates.extend(reversed(cands))

    for cand in candidates:
        if list((cand / "lightop").glob("op*.so")):
            cand_str = str(cand)
            if cand_str not in sys.path:
                sys.path.insert(0, cand_str)
                print(f"[env] Added local build to sys.path: {cand_str}")
            return


# =============================================================================
# Stage 1: Environment & Binary ISA Self-Check (rocminfo / dccobjdump / readelf)
# =============================================================================

_TARGET_LINE_RE = re.compile(
    r"(?:amdgcn-amd-amdhsa--|mcpu=|arch=)(gfx\w+)|^\s*\.amdgcn_target\s+\"([^\"]+)\""
)

_LN1P_KERNEL_RE = re.compile(
    r"layer_norm_fwd_1pass_(?:small_head_vec|small_head|vec_multi|vec|scalar)_kernel"
)

_KERNEL_SPLIT_RE = re.compile(
    r"(?=^[^\n]*layer_norm_fwd_1pass_(?:small_head_vec|small_head|vec_multi|vec|scalar)_kernel[^\n]*:|\t\.protected\t)",
    flags=re.MULTILINE,
)

# Matches trailing template args:
#   bool HAS_BIAS, bool HAS_Z, bool NORM_BEFORE_GATE, bool IS_RMS_NORM, GateActivation ACT
_MANGLED_FLAGS_RE = re.compile(
    r"Lb([01])ELb([01])ELb([01])ELb([01])EL[^E]*GateActivationE([012])E"
)
_DEMANGLED_FLAGS_RE = re.compile(
    r"\b(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*"
    r"(?:(?:at::native::)?GateActivation::)?(kNone|kSilu|kSigmoid|[012])\b"
)

_GFX946_SIGNAL_INSNS = (
    "v_dot2_f32_bf16",
    "v_dot2_f32_f16",
    "v_permlane16_swap_b32",
    "v_permlane32_swap_b32",
    "v_pk_fma_f32",
    "v_pk_add_f32",
    "v_pk_mul_f32",
    "v_cvt_pk_bf16_f32",
    "v_cvt_pk_f16_f32",
    "gfx946",
)


def _extract_arch_segments(dump_text: str) -> list[tuple[str, str]]:
    """Split multi-arch dccobjdump text into `(detected_arch_or_unknown, text)` sections."""
    lines = dump_text.splitlines(keepends=True)
    segments: list[tuple[str, list[str]]] = []
    cur_arch = "unknown"
    cur_buf: list[str] = []

    for ln in lines:
        m = _TARGET_LINE_RE.search(ln)
        if m:
            found = m.group(1) or m.group(2) or ""
            arch_m = re.search(r"gfx[0-9a-zA-Z]+", found)
            new_arch = arch_m.group(0).lower() if arch_m else cur_arch
            if cur_buf and new_arch != cur_arch:
                segments.append((cur_arch, cur_buf))
                cur_buf = []
            cur_arch = new_arch
        cur_buf.append(ln)
    if cur_buf:
        segments.append((cur_arch, cur_buf))
    return [(arch, "".join(buf)) for arch, buf in segments]


def _classify_ln1p_kernel(header: str) -> Optional[str]:
    if "layer_norm_fwd_1pass_small_head_vec_kernel" in header:
        return "small_head_vec"
    if "layer_norm_fwd_1pass_small_head_kernel" in header:
        return "small_head"
    if "layer_norm_fwd_1pass_vec_multi_kernel" in header:
        return "vec_multi"
    if "layer_norm_fwd_1pass_vec_kernel" in header:
        return "vec"
    if "layer_norm_fwd_1pass_scalar_kernel" in header:
        return "scalar"
    return None


def _parse_template_flags(header: str) -> Optional[dict[str, Any]]:
    """Parse (has_bias, has_z, norm_before_gate, is_rms_norm, act) from symbol header."""
    m = _MANGLED_FLAGS_RE.search(header)
    if m:
        return {
            "has_bias": m.group(1) == "1",
            "has_z": m.group(2) == "1",
            "norm_before_gate": m.group(3) == "1",
            "is_rms_norm": m.group(4) == "1",
            "act": int(m.group(5)),
        }
    m = _DEMANGLED_FLAGS_RE.search(header)
    if m:
        act_raw = m.group(5)
        act_map = {"kNone": 0, "kSilu": 1, "kSigmoid": 2, "0": 0, "1": 1, "2": 2}
        return {
            "has_bias": m.group(1) == "true",
            "has_z": m.group(2) == "true",
            "norm_before_gate": m.group(3) == "true",
            "is_rms_norm": m.group(4) == "true",
            "act": act_map[act_raw],
        }
    return None


def _extract_block_scratch(blk: str) -> Optional[int]:
    for m in re.finditer(
        r"(?:\.amdhsa_)?private_segment_fixed_size\s*(?:[:= ]\s*)(\d+)|SCRATCH\D+(\d+)",
        blk,
    ):
        return int(m.group(1) or m.group(2))
    return None


def _extract_block_vgpr(blk: str) -> Optional[int]:
    m = re.search(
        r"(?:\.amdhsa_next_free_vgpr|\.vgpr_count|VGPR\s*(?:count)?)\s*(?:[:= ]\s*)(\d+)",
        blk,
        flags=re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


def _is_half_or_bf16_symbol(header: str) -> bool:
    return any(tok in header for tok in ("BFloat16", "Half", "8BFloat16", "4Half"))


def audit_dccobjdump_layer_norm_1pass_isa(dump_text: str, strict: bool = False) -> str:
    """Parse dccobjdump output for gfx946 layer_norm_fwd_1pass kernels & verify ISA/resources."""
    segments = _extract_arch_segments(dump_text)
    explicit_946_present = any(arch == "gfx946" for arch, _ in segments)

    candidates: list[tuple[str, str, str]] = []  # (seg_arch, family, block_text)
    for seg_arch, seg_text in segments:
        if explicit_946_present and seg_arch != "gfx946":
            continue
        blocks = _KERNEL_SPLIT_RE.split(seg_text)
        for blk in blocks:
            if len(blk.splitlines()) <= 1:
                continue
            head = blk[:600]
            if not _LN1P_KERNEL_RE.search(head):
                continue
            fam = _classify_ln1p_kernel(head)
            if fam is not None:
                candidates.append((seg_arch, fam, blk))

    if not candidates:
        msg = "No layer_norm_fwd_1pass_* kernel blocks found in dccobjdump output"
        if strict:
            raise AssertionError(msg)
        return f"[isa][warn] {msg}"

    expected_insns = {
        "v_dot2_f32_bf16": 0,
        "v_dot2_f32_f16": 0,
        "v_pk_add_f32": 0,
        "v_pk_fma_f32": 0,
        "v_pk_mul_f32": 0,
        "v_cvt_pk_bf16_f32": 0,
        "v_cvt_pk_f16_f32": 0,
        "v_permlane16_swap_b32": 0,
        "v_permlane32_swap_b32": 0,
        "v_mov_b32_dpp": 0,
        "v_rcp_f32": 0,
        "v_exp_f32": 0,
        "ds_bpermute_b32": 0,
    }
    audited_by_family = {
        "small_head_vec": 0,
        "small_head": 0,
        "vec": 0,
        "vec_multi": 0,
        "scalar": 0,
    }
    max_vgpr_by_family: dict[str, int] = {k: 0 for k in audited_by_family}
    max_vgpr_raw_rms_fast: dict[str, int] = {
        "small_head_vec": 0,
        "vec": 0,
        "vec_multi": 0,
    }

    scratch_violations: list[str] = []
    same_reg_swaps: list[str] = []
    forbidden_rtz: list[str] = []
    forbidden_div_in_vec: list[str] = []
    forbidden_pregate_or_ln_dot2: list[str] = []
    forbidden_fp64_in_16b: list[str] = []
    forbidden_bpermute_in_vec: list[str] = []

    vec_families = ("small_head_vec", "vec", "vec_multi")

    for seg_arch, fam, blk in candidates:
        has_946_signal = (seg_arch == "gfx946") or any(kw in blk for kw in _GFX946_SIGNAL_INSNS)
        if not has_946_signal:
            continue

        audited_by_family[fam] += 1
        header_full = blk.splitlines()[0].strip()
        header = header_full[:140]
        is_16b = _is_half_or_bf16_symbol(header_full)
        flags = _parse_template_flags(header_full)

        scratch = _extract_block_scratch(blk)
        if scratch is not None and scratch != 0:
            scratch_violations.append(f"{header} (scratch={scratch})")

        vgpr = _extract_block_vgpr(blk)
        if vgpr is not None:
            if vgpr > max_vgpr_by_family[fam]:
                max_vgpr_by_family[fam] = vgpr
            if (
                fam in max_vgpr_raw_rms_fast
                and flags is not None
                and flags["is_rms_norm"]
                and (not flags["has_z"] or flags["norm_before_gate"] or flags["act"] == 0)
                and vgpr > max_vgpr_raw_rms_fast[fam]
            ):
                max_vgpr_raw_rms_fast[fam] = vgpr

        for kw in expected_insns:
            expected_insns[kw] += blk.count(kw)

        # Red line 1: Errata R4 — distinct VGPRs required for permlane swap.
        for swap_kw in ("v_permlane16_swap_b32", "v_permlane32_swap_b32"):
            for ln in blk.splitlines():
                if swap_kw in ln:
                    regs = re.findall(r"\bv\d+\b", ln)
                    if len(regs) >= 2 and regs[0] == regs[1]:
                        same_reg_swaps.append(f"{header}: {ln.strip()}")

        # Red line 2: Forbidden RTZ packed convert (must use RNE v_cvt_pk_{bf16,f16}_f32).
        if "v_cvt_pkrtz_" in blk:
            forbidden_rtz.append(f"{header} (v_cvt_pkrtz x{blk.count('v_cvt_pkrtz_')})")

        # Red line 3: Zero IEEE FP32 division in gfx946 BF16/FP16 vectorized kernels.
        if fam in vec_families and is_16b:
            for div_kw in ("v_div_scale_f32", "v_div_fmas_f32"):
                if div_kw in blk:
                    forbidden_div_in_vec.append(f"{header} ({div_kw})")
                    break

        # Red line 4: Pre-gate (!NORM_BEFORE_GATE && HAS_Z && ACT!=0) and LayerNorm
        # (!IS_RMS_NORM) must keep strict FP32 values — NEVER emit v_dot2_f32_{bf16,f16}.
        if flags is not None:
            is_pre_gate = flags["has_z"] and (not flags["norm_before_gate"]) and (flags["act"] != 0)
            is_layernorm = not flags["is_rms_norm"]
            if (is_pre_gate or is_layernorm) and (
                "v_dot2_f32_bf16" in blk or "v_dot2_f32_f16" in blk
            ):
                reason = "pre-gate" if is_pre_gate else "layernorm"
                forbidden_pregate_or_ln_dot2.append(f"{header} ({reason})")

        # Red line 5: BF16/FP16 instantiations must not silently promote to FP64 VALU.
        if is_16b:
            for f64_kw in ("v_add_f64", "v_mul_f64", "v_fma_f64", "v_max_f64"):
                if f64_kw in blk:
                    forbidden_fp64_in_16b.append(f"{header} ({f64_kw})")
                    break

        # Red line 6: Zero ds_bpermute_b32 in gfx946 BF16/FP16 vectorized kernels.
        if fam in vec_families and is_16b and "ds_bpermute_b32" in blk:
            forbidden_bpermute_in_vec.append(
                f"{header} (ds_bpermute_b32 x{blk.count('ds_bpermute_b32')})"
            )

    total_audited = sum(audited_by_family.values())
    if total_audited == 0:
        msg = (
            "Found layer_norm_fwd_1pass kernel blocks in dccobjdump, but none matched "
            "gfx946 signals (was op*.so built with gfx946?)"
        )
        if strict:
            raise AssertionError(msg)
        return f"[isa][warn] {msg}"

    if scratch_violations:
        raise AssertionError(
            "Non-zero scratch spill in layer_norm_fwd_1pass kernels: "
            + "; ".join(scratch_violations)
        )
    if same_reg_swaps:
        raise AssertionError(
            "Same-register operand in permlane swap (errata R4): "
            + "; ".join(same_reg_swaps)
        )
    if forbidden_rtz:
        raise AssertionError(
            "Forbidden RTZ v_cvt_pkrtz_ in gfx946 layer_norm_fwd_1pass kernels: "
            + "; ".join(forbidden_rtz)
        )
    if forbidden_div_in_vec:
        raise AssertionError(
            "Unexpected IEEE FP32 division in gfx946 BF16/FP16 vectorized kernels: "
            + "; ".join(forbidden_div_in_vec)
        )
    if forbidden_pregate_or_ln_dot2:
        raise AssertionError(
            "v_dot2_f32_{bf16,f16} emitted in pre-gate or LayerNorm kernel "
            "(violates FP32 gated/centered accuracy): "
            + "; ".join(forbidden_pregate_or_ln_dot2)
        )
    if forbidden_fp64_in_16b:
        raise AssertionError(
            "Unexpected FP64 VALU in BF16/FP16 kernels: "
            + "; ".join(forbidden_fp64_in_16b)
        )
    if forbidden_bpermute_in_vec:
        raise AssertionError(
            "Unexpected ds_bpermute_b32 in gfx946 BF16/FP16 vectorized kernels: "
            + "; ".join(forbidden_bpermute_in_vec)
        )

    required_insns = (
        "v_dot2_f32_bf16",
        "v_dot2_f32_f16",
        "v_pk_add_f32",
        "v_pk_fma_f32",
        "v_pk_mul_f32",
        "v_cvt_pk_bf16_f32",
        "v_cvt_pk_f16_f32",
        "v_permlane16_swap_b32",
        "v_permlane32_swap_b32",
        "v_mov_b32_dpp",
        "v_rcp_f32",
        "v_exp_f32",
    )
    required_fams = ("small_head_vec", "vec", "vec_multi")
    missing = [k for k in required_insns if expected_insns[k] == 0]
    missing_fams = [k for k in required_fams if audited_by_family[k] == 0]
    if strict and (missing or missing_fams):
        raise AssertionError(
            f"Missing expected gfx946 ISA hits={missing} or kernel families={missing_fams}"
        )

    lines = [
        f"[isa] Audited {total_audited} gfx946 layer_norm_fwd_1pass kernel section(s) "
        "(SCRATCH=0, distinct permlane regs, zero RTZ/IEEE-div/pregate-dot2/FP64/vec-bpermute):"
    ]
    for fam, cnt in audited_by_family.items():
        vgpr_str = f"maxVGPR={max_vgpr_by_family[fam]}" if max_vgpr_by_family[fam] > 0 else "maxVGPR=N/A"
        extra = ""
        if fam in max_vgpr_raw_rms_fast and max_vgpr_raw_rms_fast[fam] > 0:
            rf = max_vgpr_raw_rms_fast[fam]
            flag = " [<=64 target OK]" if rf <= 64 else " [WARN >64 VGPR]"
            extra = f", raw_rms_fast maxVGPR={rf}{flag}"
        lines.append(f"  family {fam:<16}: {cnt:>4} section(s) ({vgpr_str}{extra})")
    for kw, cnt in expected_insns.items():
        lines.append(f"  {kw:<24}: {cnt} hit(s)")
    if missing or missing_fams:
        lines.append(
            f"  [warn] Missing instructions={missing}, families={missing_fams} "
            "(may need per-arch extraction or gfx946 build)"
        )
    return "\n".join(lines)


def stage_env_and_isa_check(args: argparse.Namespace) -> dict[str, object]:
    print("\n================ 1. ENVIRONMENT & GFX946 ISA CHECK ================")
    info: dict[str, object] = {
        "arch": "cpu",
        "cu_count": 0,
        "op_so": None,
    }

    for tool in ("hy-smi", "rocm-smi"):
        if shutil.which(tool):
            rc, out = run_cmd([tool], timeout=20)
            if rc == 0:
                print(f"[{tool}]\n" + "\n".join(out.splitlines()[:20]))
                break

    if shutil.which("rocminfo"):
        rc, out = run_cmd(["rocminfo"], timeout=20)
        if rc == 0:
            interesting = [
                line.strip()
                for line in out.splitlines()
                if re.search(r"(Name:|Compute Unit|Marketing Name|gfx)", line, re.IGNORECASE)
            ]
            print("[rocminfo summary]\n  " + "\n  ".join(interesting[:16]))

    print(f"[python] {sys.executable} ({sys.version.split()[0]})")
    if torch is not None:
        print(f"[torch]  {torch.__version__} (hip={torch.version.hip}, cuda_avail={torch.cuda.is_available()})")

    offline_isa_mode = bool(args.isa_only and (args.so_path or getattr(args, "isa_dump", None)))

    if torch is not None and torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        arch = props.gcnArchName.split(":", 1)[0]
        info["arch"] = arch
        info["cu_count"] = props.multi_processor_count
        print(
            f"[device {dev}] name={props.name}, gcnArchName={props.gcnArchName}, "
            f"CUs={props.multi_processor_count}, total_mem={props.total_memory / (1024**3):.1f} GiB"
        )
        if args.require_gfx946 and arch != "gfx946":
            raise RuntimeError(f"--require-gfx946 set, but active GPU arch is {arch!r}")
        if arch != "gfx946":
            print(
                f"[warn] Current live GPU is {arch}, not gfx946 "
                "(kernel execution tests that arch; op*.so gfx946 ISA is still audited)."
            )
    elif not offline_isa_mode:
        raise RuntimeError("No CUDA/ROCm device (or PyTorch) available.")

    if args.skip_isa:
        print("[isa] Skipped by --skip-isa")
        return info

    isa_dump = getattr(args, "isa_dump", None)
    if isa_dump:
        dump_path = Path(isa_dump).expanduser()
        if not dump_path.is_file():
            raise FileNotFoundError(f"--isa-dump file not found: {dump_path}")
        print(f"[isa] Auditing offline disassembly dump: {dump_path}")
        print(audit_dccobjdump_layer_norm_1pass_isa(dump_path.read_text(encoding="utf-8"), strict=args.strict_isa))
        return info

    so_path = args.so_path
    if not so_path:
        import lightop
        from lightop import op as lightop_native

        so_path = getattr(lightop_native, "__file__", None)
        print(f"[lightop] pkg={lightop.__file__}")
    info["op_so"] = so_path
    print(f"[lightop] op_so={so_path}")

    if not so_path or not os.path.exists(so_path):
        msg = f"Compiled extension op*.so not found: {so_path}"
        if args.strict_isa:
            raise FileNotFoundError(msg)
        print(f"[isa][warn] {msg}")
        return info

    if shutil.which("readelf"):
        rc_re, out_re = run_cmd(["readelf", "--dyn-syms", "--wide", so_path], timeout=30)
        if rc_re == 0:
            bad = [ln for ln in out_re.splitlines() if "__extendhfsf2" in ln and " UND " in ln]
            if bad:
                raise AssertionError("Unresolved __extendhfsf2 in .dynsym:\n" + "\n".join(bad))
            print("[readelf] OK: no unresolved __extendhfsf2 in .dynsym")

    if shutil.which("hipprof"):
        rc_ca, out_ca = run_cmd(["hipprof", "--codeobj-analyze", so_path], timeout=120)
        if rc_ca == 0:
            ln_lines = [
                ln.strip()
                for ln in out_ca.splitlines()
                if any(k in ln for k in ("layer_norm_fwd_1pass", "gfx946"))
            ]
            if ln_lines:
                print("[hipprof --codeobj-analyze excerpt]\n  " + "\n  ".join(ln_lines[:24]))

    dccobjdump = shutil.which("dccobjdump") or (
        "/opt/dtk/bin/dccobjdump" if os.path.exists("/opt/dtk/bin/dccobjdump") else None
    )
    if not dccobjdump:
        msg = "dccobjdump not found in PATH or /opt/dtk/bin"
        if args.strict_isa:
            raise RuntimeError(msg)
        print(f"[isa][warn] {msg}")
        return info

    cmd = [
        dccobjdump,
        f"--inputs={so_path}",
        "--show-sass",
        "--show-instruction-encoding",
        "--separate-functions",
    ]
    print(f"[isa] Running: {' '.join(cmd)}")
    rc_d, out_d = run_cmd(cmd, timeout=240)
    if rc_d != 0:
        msg = f"dccobjdump exited with {rc_d}: {out_d[:500]}"
        if args.strict_isa:
            raise RuntimeError(msg)
        print(f"[isa][warn] {msg}")
        return info

    print(audit_dccobjdump_layer_norm_1pass_isa(out_d, strict=args.strict_isa))
    return info


# =============================================================================
# Independent PyTorch FP32 Reference & Call Helper
# =============================================================================

def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def ref_layer_norm_fwd_1pass(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    *,
    m: int,
    n: int,
    groups: int,
    eps: float,
    has_bias: bool,
    has_z: bool,
    norm_before_gate: bool,
    is_rms_norm: bool,
    activation: str,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    total_cols = groups * n
    x_f = x[:m, :total_cols].float().reshape(m, groups, n)
    w_f = w[:total_cols].float().reshape(1, groups, n)
    b_f = b[:total_cols].float().reshape(1, groups, n) if has_bias and b is not None else None
    z_f = z[:m, :total_cols].float().reshape(m, groups, n) if has_z and z is not None else None

    active_gate = has_z and activation not in ("", "none", "identity")
    if active_gate and not norm_before_gate:
        assert z_f is not None
        gate = z_f * torch.sigmoid(z_f) if activation in ("swish", "silu") else torch.sigmoid(z_f)
        x_f = x_f * gate

    if is_rms_norm:
        mean_out = None
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = x_f * rstd
    else:
        mean = x_f.mean(dim=-1, keepdim=True)
        diff = x_f - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = diff * rstd
        mean_out = mean.squeeze(-1).transpose(0, 1).contiguous()

    y_f = x_hat * w_f
    if has_bias and b_f is not None:
        y_f = y_f + b_f
    if active_gate and norm_before_gate:
        assert z_f is not None
        gate = z_f * torch.sigmoid(z_f) if activation in ("swish", "silu") else torch.sigmoid(z_f)
        y_f = y_f * gate

    return y_f.reshape(m, total_cols).to(x.dtype), mean_out, rstd.squeeze(-1).transpose(0, 1).contiguous()


def invoke_ln1p(
    fn,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    mean: Optional[torch.Tensor],
    rstd: torch.Tensor,
    *,
    m: int,
    n: int,
    eps: float,
    has_bias: bool,
    has_z: bool,
    norm_before_gate: bool,
    is_rms_norm: bool,
    activation: str,
    stride_x_row: Optional[int] = None,
    stride_y_row: Optional[int] = None,
    stride_z_row: Optional[int] = None,
) -> None:
    sx = x.stride(0) if stride_x_row is None else stride_x_row
    sy = y.stride(0) if stride_y_row is None else stride_y_row
    sz = (z.stride(0) if (has_z and z is not None) else 0) if stride_z_row is None else stride_z_row
    fn(
        x,
        y,
        w,
        b,
        z,
        mean,
        rstd,
        sx,
        sy,
        sz,
        m,
        n,
        eps,
        _next_pow2(n),
        1,
        has_bias,
        has_z,
        norm_before_gate,
        is_rms_norm,
        activation,
    )


def assert_matches_ref(
    x_ref: torch.Tensor,
    y_act: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    mean_act: Optional[torch.Tensor],
    rstd_act: torch.Tensor,
    *,
    m: int,
    n: int,
    groups: int,
    eps: float,
    has_bias: bool,
    has_z: bool,
    norm_before_gate: bool,
    is_rms_norm: bool,
    activation: str,
    tag: str,
) -> float:
    ref_y, ref_mean, ref_rstd = ref_layer_norm_fwd_1pass(
        x_ref, w, b, z,
        m=m, n=n, groups=groups, eps=eps,
        has_bias=has_bias, has_z=has_z,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm, activation=activation,
    )
    assert torch.isfinite(y_act).all(), f"[{tag}] non-finite Y"
    assert torch.isfinite(rstd_act).all(), f"[{tag}] non-finite Rstd"
    if x_ref.dtype == torch.bfloat16:
        atol_y, rtol_y, atol_s, rtol_s, l2_tol = 3e-2, 3e-2, 1e-2, 1e-2, 1.5e-2
    elif x_ref.dtype == torch.float16:
        atol_y, rtol_y, atol_s, rtol_s, l2_tol = 5e-3, 5e-3, 2e-3, 2e-3, 2.5e-3
    else:
        atol_y, rtol_y, atol_s, rtol_s, l2_tol = 1e-4, 1e-4, 1e-5, 1e-5, 1e-4

    torch.testing.assert_close(y_act.float(), ref_y.float(), atol=atol_y, rtol=rtol_y, msg=f"[{tag}] Y mismatch")
    torch.testing.assert_close(rstd_act.reshape(groups, m), ref_rstd, atol=atol_s, rtol=rtol_s, msg=f"[{tag}] Rstd mismatch")
    if not is_rms_norm:
        assert mean_act is not None and ref_mean is not None
        torch.testing.assert_close(mean_act.reshape(groups, m), ref_mean, atol=atol_s, rtol=rtol_s, msg=f"[{tag}] Mean mismatch")

    ref_norm = ref_y.float().norm().item()
    rel_l2 = ((y_act.float() - ref_y.float()).norm().item() / ref_norm) if ref_norm > 1e-4 else 0.0
    assert rel_l2 < l2_tol, f"[{tag}] rel_l2={rel_l2:.6e} >= {l2_tol}"
    return rel_l2


# =============================================================================
# Stage 2: Full Boundary Accuracy, Mamba2 Strided Gate & 4-XCD Consistency Stress
# =============================================================================

def stage_correctness_and_edge_cases(args: argparse.Namespace) -> None:
    print("\n================ 2. CORRECTNESS, STRIDED GATE & 4-XCD STRESS ================")
    from lightop import op as lightop_native
    from lightop.norm import layer_norm_fwd_1pass_opt

    eps = 1e-6

    # 2.1 Zero-API-change binding parity & invalid-input rejection checks
    assert layer_norm_fwd_1pass_opt is lightop_native.layer_norm_fwd_1pass_opt
    x_chk = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    y_chk1 = torch.empty_like(x_chk)
    y_chk2 = torch.empty_like(x_chk)
    w_chk = torch.randn((128,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
    z_chk = torch.randn_like(x_chk)
    r_chk1 = torch.empty((1, 8), device="cuda", dtype=torch.float32)
    r_chk2 = torch.empty((1, 8), device="cuda", dtype=torch.float32)
    invoke_ln1p(
        layer_norm_fwd_1pass_opt, x_chk, y_chk1, w_chk, None, z_chk, None, r_chk1,
        m=8, n=128, eps=eps, has_bias=False, has_z=True,
        norm_before_gate=True, is_rms_norm=True, activation="swish",
    )
    invoke_ln1p(
        lightop_native.layer_norm_fwd_1pass_opt, x_chk, y_chk2, w_chk, None, z_chk, None, r_chk2,
        m=8, n=128, eps=eps, has_bias=False, has_z=True,
        norm_before_gate=True, is_rms_norm=True, activation="swish",
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(y_chk1, y_chk2, rtol=0, atol=0)
    torch.testing.assert_close(r_chk1, r_chk2, rtol=0, atol=0)

    # Storage-capacity overflow & non-contiguous last-dim must raise RuntimeError
    for neg_tag, bad_z, bad_sz in (
        ("storage_overflow", z_chk, 128 + 16),
        ("non_contig_last_dim", torch.randn((128, 8), device="cuda", dtype=torch.bfloat16).t(), 128),
    ):
        raised = False
        try:
            invoke_ln1p(
                layer_norm_fwd_1pass_opt, x_chk, y_chk1, w_chk, None, bad_z, None, r_chk1,
                m=8, n=128, eps=eps, has_bias=False, has_z=True,
                norm_before_gate=True, is_rms_norm=True, activation="swish",
                stride_z_row=bad_sz,
            )
        except RuntimeError:
            raised = True
        assert raised, f"Expected RuntimeError for invalid input: {neg_tag}"
    print("  [PASS] Zero-API-change binding parity & invalid-storage/layout guards")

    torch.manual_seed(946)

    # 2.2 Full 5-family dispatch matrix (small_head_vec, small_head, vec, vec_multi, scalar)
    hiddens = (
        (64, 128, 256, 512, 1024, 4096, 8192, 65, 520)
        if args.quick
        else (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192, 16384, 33, 65, 520, 16392)
    )
    cases_run = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for n in hiddens:
            modes = (
                ((True, False, True, True, "swish"), (True, False, False, True, "swish"))
                if (args.quick and dtype == torch.float32)
                else (
                    (True, False, True, True, "swish"),
                    (True, True, True, True, "sigmoid"),
                    (True, False, False, True, "swish"),
                    (False, True, True, False, "swish"),
                    (True, False, False, False, "none"),
                )
            )
            for is_rms, has_bias, nbg, has_z, act in modes:
                m = 17
                x = torch.randn((m, n), device="cuda", dtype=dtype) * 0.5
                y = torch.empty_like(x)
                w = torch.randn((n,), device="cuda", dtype=dtype) * 0.1 + 1.0
                b = torch.randn((n,), device="cuda", dtype=dtype) * 0.1 if has_bias else None
                z = torch.randn((m, n), device="cuda", dtype=dtype) * 0.5 if has_z else None
                mean = None if is_rms else torch.empty((1, m), device="cuda", dtype=torch.float32)
                rstd = torch.empty((1, m), device="cuda", dtype=torch.float32)
                invoke_ln1p(
                    layer_norm_fwd_1pass_opt, x, y, w, b, z, mean, rstd,
                    m=m, n=n, eps=eps, has_bias=has_bias, has_z=has_z,
                    norm_before_gate=nbg, is_rms_norm=is_rms, activation=act,
                )
                torch.cuda.synchronize()
                assert_matches_ref(
                    x, y, w, b, z, mean, rstd,
                    m=m, n=n, groups=1, eps=eps,
                    has_bias=has_bias, has_z=has_z,
                    norm_before_gate=nbg, is_rms_norm=is_rms, activation=act,
                    tag=f"mat_{dtype}_N{n}_rms={is_rms}_nbg={nbg}_z={has_z}_{act}",
                )
                cases_run += 1
    print(f"  [PASS] 5-family dispatch & mode matrix ({cases_run} cases)")

    # 2.3 Mamba2 strided gate views (z_vec_mode in {1, 2, 4, 8}, non-zero storage_offset, end-split, strided X/Y)
    strided_cases = 0
    for dtype in (torch.bfloat16, torch.float16):
        for n, groups in ((128, 1), (256, 4), (512, 4), (1024, 2), (4096, 1)):
            for pad_z in (1, 2, 4, 12, 16):
                for z_off in (0, 7, 16):
                    for nbg in (False, True):
                        m = 19
                        total_cols = groups * n
                        x = torch.randn((m, total_cols), device="cuda", dtype=dtype) * 0.5
                        y = torch.empty_like(x)
                        w = torch.randn((total_cols,), device="cuda", dtype=dtype) * 0.1 + 1.0
                        z_full = torch.randn((m, z_off + total_cols + pad_z), device="cuda", dtype=dtype) * 0.5
                        z = z_full[:, z_off : z_off + total_cols]
                        rstd = torch.empty((groups, m), device="cuda", dtype=torch.float32)
                        invoke_ln1p(
                            layer_norm_fwd_1pass_opt, x, y, w, None, z, None, rstd,
                            m=m, n=n, eps=eps, has_bias=False, has_z=True,
                            norm_before_gate=nbg, is_rms_norm=True, activation="swish",
                        )
                        torch.cuda.synchronize()
                        assert_matches_ref(
                            x, y, w, None, z, None, rstd,
                            m=m, n=n, groups=groups, eps=eps,
                            has_bias=False, has_z=True, norm_before_gate=nbg,
                            is_rms_norm=True, activation="swish",
                            tag=f"mamba2_{dtype}_N{n}_G{groups}_pad{pad_z}_off{z_off}_nbg={nbg}",
                        )
                        strided_cases += 1

    # Exact end-of-buffer torch.split + strided Y padding guard
    m_sp, n_sp, g_sp = 32, 256, 4
    tc_sp = n_sp * g_sp
    proj = torch.randn((m_sp, tc_sp + 549), device="cuda", dtype=torch.bfloat16) * 0.5
    _, z_last = torch.split(proj, [549, tc_sp], dim=-1)
    x_sp = torch.randn((m_sp, tc_sp + 16), device="cuda", dtype=torch.bfloat16)[:, :tc_sp] * 0.5
    y_full = torch.full((m_sp, tc_sp + 16), 7.5, device="cuda", dtype=torch.bfloat16)
    y_sp = y_full[:, :tc_sp]
    w_sp = torch.randn((tc_sp,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
    r_sp = torch.empty((g_sp, m_sp), device="cuda", dtype=torch.float32)
    invoke_ln1p(
        layer_norm_fwd_1pass_opt, x_sp, y_sp, w_sp, None, z_last, None, r_sp,
        m=m_sp, n=n_sp, eps=eps, has_bias=False, has_z=True,
        norm_before_gate=False, is_rms_norm=True, activation="swish",
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(y_full[:, tc_sp:], torch.full((m_sp, 16), 7.5, device="cuda", dtype=torch.bfloat16), rtol=0, atol=0)
    assert_matches_ref(
        x_sp, y_sp, w_sp, None, z_last, None, r_sp,
        m=m_sp, n=n_sp, groups=g_sp, eps=eps,
        has_bias=False, has_z=True, norm_before_gate=False,
        is_rms_norm=True, activation="swish", tag="mamba2_end_split_strided_xy",
    )
    strided_cases += 1
    print(f"  [PASS] Mamba2 strided gate, end-split & strided X/Y regression ({strided_cases} cases)")

    # 2.4 Hardware & numerical edge cases: extreme z, LayerNorm large-mean cancellation,
    #     zero/-0.0 rows, in-place, unaligned offset=1, M=0, M=288, HIP Graph replay.
    for dtype in (torch.bfloat16, torch.float16):
        for n in (64, 128, 256, 512, 1024, 4096):
            # Extreme gate + signed zero + all-zero row
            m = 8
            x = torch.randn((m, n), device="cuda", dtype=dtype) * 0.5
            x[1].zero_()
            x[2, 0] = -0.0
            y = torch.empty_like(x)
            w = torch.randn((n,), device="cuda", dtype=dtype) * 0.1 + 1.0
            z = torch.randn((m, n), device="cuda", dtype=dtype) * 0.5
            pat = torch.tensor([-100.0, -20.0, -1e-4, 0.0, 1e-4, 20.0, 100.0], device="cuda", dtype=dtype)
            z[0].copy_(pat.repeat((n + 6) // 7)[:n])
            rstd = torch.empty((1, m), device="cuda", dtype=torch.float32)
            for nbg in (True, False):
                invoke_ln1p(
                    layer_norm_fwd_1pass_opt, x, y, w, None, z, None, rstd,
                    m=m, n=n, eps=eps, has_bias=False, has_z=True,
                    norm_before_gate=nbg, is_rms_norm=True, activation="swish",
                )
                torch.cuda.synchronize()
                assert_matches_ref(
                    x, y, w, None, z, None, rstd,
                    m=m, n=n, groups=1, eps=eps,
                    has_bias=False, has_z=True, norm_before_gate=nbg,
                    is_rms_norm=True, activation="swish", tag=f"extz_{dtype}_N{n}_nbg={nbg}",
                )

            # LayerNorm large-mean cancellation (tests P0 2-pass register variance)
            x_ln = torch.randn((17, n), device="cuda", dtype=dtype) * 0.5 + 1000.0
            y_ln = torch.empty_like(x_ln)
            b_ln = torch.randn((n,), device="cuda", dtype=dtype) * 0.1
            mean_ln = torch.empty((1, 17), device="cuda", dtype=torch.float32)
            rstd_ln = torch.empty((1, 17), device="cuda", dtype=torch.float32)
            invoke_ln1p(
                layer_norm_fwd_1pass_opt, x_ln, y_ln, w, b_ln, None, mean_ln, rstd_ln,
                m=17, n=n, eps=eps, has_bias=True, has_z=False,
                norm_before_gate=False, is_rms_norm=False, activation="none",
            )
            torch.cuda.synchronize()
            assert_matches_ref(
                x_ln, y_ln, w, b_ln, None, mean_ln, rstd_ln,
                m=17, n=n, groups=1, eps=eps,
                has_bias=True, has_z=False, norm_before_gate=False,
                is_rms_norm=False, activation="none", tag=f"ln_cancel_{dtype}_N{n}",
            )

    # In-place Y is X, unaligned offset=1, empty M=0
    for n in (128, 512, 4096, 520):
        x_ip = torch.randn((17, n), device="cuda", dtype=torch.bfloat16) * 0.5
        x_ip0 = x_ip.clone()
        w_ip = torch.randn((n,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        z_ip = torch.randn_like(x_ip)
        r_ip = torch.empty((1, 17), device="cuda", dtype=torch.float32)
        invoke_ln1p(
            layer_norm_fwd_1pass_opt, x_ip, x_ip, w_ip, None, z_ip, None, r_ip,
            m=17, n=n, eps=eps, has_bias=False, has_z=True,
            norm_before_gate=True, is_rms_norm=True, activation="swish",
        )
        torch.cuda.synchronize()
        assert_matches_ref(
            x_ip0, x_ip, w_ip, None, z_ip, None, r_ip,
            m=17, n=n, groups=1, eps=eps, has_bias=False, has_z=True,
            norm_before_gate=True, is_rms_norm=True, activation="swish", tag=f"inplace_N{n}",
        )

        x_u = (torch.randn((17 * n + 1,), device="cuda", dtype=torch.bfloat16) * 0.5)[1:].view(17, n)
        y_u = torch.empty((17 * n + 1,), device="cuda", dtype=torch.bfloat16)[1:].view(17, n)
        w_u = (torch.randn((n + 1,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0)[1:]
        z_u = (torch.randn((17 * n + 1,), device="cuda", dtype=torch.bfloat16) * 0.5)[1:].view(17, n)
        invoke_ln1p(
            layer_norm_fwd_1pass_opt, x_u, y_u, w_u, None, z_u, None, r_ip,
            m=17, n=n, eps=eps, has_bias=False, has_z=True,
            norm_before_gate=False, is_rms_norm=True, activation="swish",
        )
        torch.cuda.synchronize()
        assert_matches_ref(
            x_u, y_u, w_u, None, z_u, None, r_ip,
            m=17, n=n, groups=1, eps=eps, has_bias=False, has_z=True,
            norm_before_gate=False, is_rms_norm=True, activation="swish", tag=f"unaligned_N{n}",
        )

        xe = torch.empty((0, n), device="cuda", dtype=torch.bfloat16)
        ye = torch.empty_like(xe)
        ze = torch.empty_like(xe)
        re_ = torch.empty((0,), device="cuda", dtype=torch.float32)
        invoke_ln1p(
            layer_norm_fwd_1pass_opt, xe, ye, w_ip, None, ze, None, re_,
            m=0, n=n, eps=eps, has_bias=False, has_z=True,
            norm_before_gate=True, is_rms_norm=True, activation="swish",
        )
    torch.cuda.synchronize()

    # M=288 (128-CU vs 192-CU rule) & HIP Graph capture/replay
    for n, groups, pad_z, nbg in ((128, 1, 0, True), (512, 4, 12, False), (4096, 1, 0, True)):
        m = 288
        total_cols = groups * n
        x = torch.randn((m, total_cols), device="cuda", dtype=torch.bfloat16) * 0.5
        y = torch.empty_like(x)
        w = torch.randn((total_cols,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        z_full = torch.randn((m, total_cols + pad_z), device="cuda", dtype=torch.bfloat16) * 0.5
        z = z_full[:, :total_cols]
        rstd = torch.empty((groups, m), device="cuda", dtype=torch.float32)
        x_src, z_src = x.clone(), z_full.clone()
        invoke_ln1p(
            layer_norm_fwd_1pass_opt, x, y, w, None, z, None, rstd,
            m=m, n=n, eps=eps, has_bias=False, has_z=True,
            norm_before_gate=nbg, is_rms_norm=True, activation="swish",
        )
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            x.copy_(x_src)
            z_full.copy_(z_src)
            invoke_ln1p(
                layer_norm_fwd_1pass_opt, x, y, w, None, z, None, rstd,
                m=m, n=n, eps=eps, has_bias=False, has_z=True,
                norm_before_gate=nbg, is_rms_norm=True, activation="swish",
            )
        for rep in range(3):
            x_src.normal_(std=0.5)
            z_src.normal_(std=0.5)
            g.replay()
            torch.cuda.synchronize()
            assert_matches_ref(
                x_src, y, w, None, z_src[:, :total_cols], None, rstd,
                m=m, n=n, groups=groups, eps=eps,
                has_bias=False, has_z=True, norm_before_gate=nbg,
                is_rms_norm=True, activation="swish", tag=f"graph_M288_N{n}_r{rep}",
            )
    print("  [PASS] Extreme gate, LayerNorm cancellation, in-place, unaligned, M=0/288 & HIP Graph")

    # 2.5 Large-batch int64 indexing stress
    if (not args.quick or args.include_large_stress) and args.stress_tokens > 0:
        st = args.stress_tokens
        for n in (128, 4096):
            x_big = torch.randn((st, n), device="cuda", dtype=torch.bfloat16) * 0.5
            y_big = torch.empty_like(x_big)
            w_big = torch.randn((n,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
            z_big = torch.randn((st, n), device="cuda", dtype=torch.bfloat16) * 0.5
            rstd_big = torch.empty((1, st), device="cuda", dtype=torch.float32)
            invoke_ln1p(
                layer_norm_fwd_1pass_opt, x_big, y_big, w_big, None, z_big, None, rstd_big,
                m=st, n=n, eps=eps, has_bias=False, has_z=True,
                norm_before_gate=True, is_rms_norm=True, activation="swish",
            )
            torch.cuda.synchronize()
            for sl in (slice(0, 16), slice(st - 16, st)):
                assert_matches_ref(
                    x_big[sl], y_big[sl], w_big, None, z_big[sl], None, rstd_big[:, sl],
                    m=16, n=n, groups=1, eps=eps,
                    has_bias=False, has_z=True, norm_before_gate=True,
                    is_rms_norm=True, activation="swish", tag=f"stress_N{n}",
                )
            del x_big, y_big, z_big, rstd_big
        print(f"  [PASS] Large-batch int64 stress (M={st}, N=128/4096)")

    # 2.6 4-XCD multi-block determinism & skewed-row stress
    xcd_m = 2048 if args.quick else 4096
    xcd_loops = args.xcd_loops
    for n, groups, pad_z, z_off, nbg in (
        (128, 1, 0, 0, True),    # Qwen3.5 small_head_vec
        (512, 4, 12, 7, False),  # Mamba2 skewed unaligned strided pre-gate small_head_vec
        (4096, 1, 0, 0, True),   # Large-N vec_multi
    ):
        total_cols = groups * n
        x_base = torch.randn((xcd_m, total_cols), device="cuda", dtype=torch.bfloat16) * 0.5
        z_full = torch.randn((xcd_m, z_off + total_cols + pad_z), device="cuda", dtype=torch.bfloat16) * 0.5
        z_base = z_full[:, z_off : z_off + total_cols]
        w_base = torch.randn((total_cols,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        # Inject skewed rows across all 4 XCDs (192 CUs): zeros, extreme gates, outliers
        x_base[0::64].zero_()
        z_base[1::64].fill_(-40.0)
        z_base[2::64].fill_(40.0)
        x_base[3::64] *= 50.0

        y_ref_run = torch.empty_like(x_base)
        rstd_ref_run = torch.empty((groups, xcd_m), device="cuda", dtype=torch.float32)
        invoke_ln1p(
            layer_norm_fwd_1pass_opt, x_base, y_ref_run, w_base, None, z_base, None, rstd_ref_run,
            m=xcd_m, n=n, eps=eps, has_bias=False, has_z=True,
            norm_before_gate=nbg, is_rms_norm=True, activation="swish",
        )
        torch.cuda.synchronize()
        assert_matches_ref(
            x_base[:128], y_ref_run[:128], w_base, None, z_base[:128], None, rstd_ref_run[:, :128],
            m=128, n=n, groups=groups, eps=eps,
            has_bias=False, has_z=True, norm_before_gate=nbg,
            is_rms_norm=True, activation="swish", tag=f"4xcd_ref_N{n}_G{groups}",
        )

        y_buf = torch.empty_like(x_base)
        rstd_buf = torch.empty_like(rstd_ref_run)
        for it in range(xcd_loops):
            invoke_ln1p(
                layer_norm_fwd_1pass_opt, x_base, y_buf, w_base, None, z_base, None, rstd_buf,
                m=xcd_m, n=n, eps=eps, has_bias=False, has_z=True,
                norm_before_gate=nbg, is_rms_norm=True, activation="swish",
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(y_buf, y_ref_run, rtol=0, atol=0, msg=f"[4xcd Y N={n} it={it}]")
            torch.testing.assert_close(rstd_buf, rstd_ref_run, rtol=0, atol=0, msg=f"[4xcd Rstd N={n} it={it}]")
    print(f"  [PASS] 4-XCD determinism & skewed-row stress ({xcd_loops}x iters, M={xcd_m}, N=128/512-strided/4096)")


# =============================================================================
# Stage 3: Benchmark (Decode / Prefill / Skewed, LightOp vs Triton vs TorchRef)
# =============================================================================

def _try_build_triton_ln1p():
    """Optional SGLang-equivalent Triton baseline (auto-degrades if Triton absent)."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def _triton_ln1p_kernel(
        X,
        Y,
        W,
        B,
        Z,
        Mean,
        Rstd,
        stride_x_row,
        stride_y_row,
        stride_z_row,
        M,
        N: tl.constexpr,
        eps,
        BLOCK_N: tl.constexpr,
        ROWS_PER_BLOCK: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_Z: tl.constexpr,
        NORM_BEFORE_GATE: tl.constexpr,
        IS_RMS_NORM: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        row_start = tl.program_id(0) * ROWS_PER_BLOCK
        group = tl.program_id(1)
        rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
        cols = tl.arange(0, BLOCK_N)
        col_offsets = cols[None, :] + group * N
        row_mask = rows[:, None] < M
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask

        x = tl.load(X + rows[:, None] * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + rows[:, None] * stride_z_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                x *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                x *= tl.sigmoid(z)

        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N
            tl.store(Mean + group * M + rows, mean, mask=rows < M)
            xbar = tl.where(mask, x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
        else:
            xbar = tl.where(mask, x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
            mean = 0.0

        rstd = tl.rsqrt(var + eps)
        tl.store(Rstd + group * M + rows, rstd, mask=rows < M)

        w_offsets = cols + group * N
        w = tl.load(W + w_offsets, mask=cols < N, other=0.0).to(tl.float32)
        x_hat = x * rstd[:, None] if IS_RMS_NORM else (x - mean[:, None]) * rstd[:, None]
        if HAS_BIAS:
            b = tl.load(B + w_offsets, mask=cols < N, other=0.0).to(tl.float32)
            y = x_hat * w[None, :] + b[None, :]
        else:
            y = x_hat * w[None, :]

        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + rows[:, None] * stride_z_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                y *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                y *= tl.sigmoid(z)

        tl.store(Y + rows[:, None] * stride_y_row + col_offsets, y, mask=mask)

    def run_triton(
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        mean: Optional[torch.Tensor],
        rstd: torch.Tensor,
        *,
        m: int,
        n: int,
        groups: int,
        eps: float,
        has_bias: bool,
        has_z: bool,
        norm_before_gate: bool,
        is_rms_norm: bool,
        activation: str,
    ) -> None:
        block_n = _next_pow2(n)
        num_warps = 1 if block_n <= 256 else (4 if block_n <= 2048 else 8)
        b_arg = b if (has_bias and b is not None) else w
        z_arg = z if (has_z and z is not None) else x
        mean_arg = mean if (not is_rms_norm and mean is not None) else rstd
        _triton_ln1p_kernel[(m, groups)](
            x,
            y,
            w,
            b_arg,
            z_arg,
            mean_arg,
            rstd,
            x.stride(0),
            y.stride(0),
            z.stride(0) if (has_z and z is not None) else 0,
            m,
            n,
            eps,
            BLOCK_N=block_n,
            ROWS_PER_BLOCK=1,
            HAS_BIAS=has_bias,
            HAS_Z=has_z,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            ACTIVATION=activation,
            num_warps=num_warps,
        )

    return run_triton


def measure_call_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def measure_copy_bandwidth_gbs(warmup: int = 10, iters: int = 50) -> Optional[float]:
    """Calibrate practical D2D HBM copy bandwidth ceiling (Read + Write = 2x buffer)."""
    try:
        elems = 32 * 1024 * 1024  # 128 MiB per tensor (256 MiB traffic)
        src = torch.empty((elems,), device="cuda", dtype=torch.float32)
        dst = torch.empty((elems,), device="cuda", dtype=torch.float32)
        src.fill_(1.0)
        us = measure_call_us(lambda: dst.copy_(src), warmup, iters)
        return (2 * elems * 4) / (us * 1e-6) / 1e9
    except RuntimeError:
        return None


def stage_benchmark(args: argparse.Namespace) -> None:
    print("\n================ 3. BENCHMARK: Decode / Prefill / Skewed Gated RMSNorm ================")
    from lightop.norm import layer_norm_fwd_1pass_opt

    eps = 1e-6
    triton_fn = None if getattr(args, "skip_triton", False) else _try_build_triton_ln1p()
    if triton_fn is None:
        print("[bench] Triton baseline disabled or unavailable; comparing LightOp vs PyTorch FP32 ref.")
    else:
        print("[bench] Triton baseline enabled (SGLang _layer_norm_fwd_1pass_kernel parity).")

    copy_bw_gbs = measure_copy_bandwidth_gbs(
        warmup=min(args.warmup, 10), iters=min(args.iters, 50)
    )
    if copy_bw_gbs is not None:
        print(f"[bench] Calibrated D2D HBM Copy Bandwidth Ceiling: {copy_bw_gbs:.2f} GB/s")

    # (regime, m, n, groups, dtype, is_rms, has_bias, has_z, nbg, act, pad_z, z_off, skewed)
    bench_cases = [
        # --- Decode (small/medium M) ---
        ("Decode",      1,     128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Decode",      16,    128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Decode",      256,   128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Decode",      256,   256,  4, torch.bfloat16, True,  False, True, False, "swish",  12, 0, False),
        ("Decode",      32,    4096, 1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Decode",      288,   4096, 1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        # --- Prefill (Qwen3.5 / SmallHead / Mamba2 Strided / LargeN / LayerNorm) ---
        ("Prefill",     1024,  128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     16384, 128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     65536, 128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     16384, 128,  1, torch.bfloat16, True,  False, True, True,  "sigmoid", 0, 0, False),
        ("Prefill",     4096,  64,   1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  256,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  512,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  256,  4, torch.bfloat16, True,  False, True, False, "swish",  12, 0, False),
        ("Prefill",     4096,  512,  4, torch.bfloat16, True,  False, True, False, "swish",  12, 0, False),
        ("Prefill",     4096,  512,  4, torch.bfloat16, True,  False, True, False, "swish",  16, 0, False),
        ("Prefill",     4096,  1024, 2, torch.bfloat16, True,  False, True, False, "swish",  12, 0, False),
        ("Prefill",     4096,  2048, 1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  4096, 1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  4096, 1, torch.float16,  True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  8192, 1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  128,  1, torch.bfloat16, False, True,  True, True,  "swish",   0, 0, False),
        ("Prefill",     4096,  4096, 1, torch.bfloat16, False, True,  True, True,  "swish",   0, 0, False),
        # --- Skewed / Imbalanced (zeros, extreme +-40 gate, 50x outliers, unaligned Mamba2, odd N) ---
        ("Skewed",      4096,  128,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, True),
        ("Skewed",      4096,  512,  4, torch.bfloat16, True,  False, True, False, "swish",  12, 7, True),
        ("Skewed",      4096,  520,  1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, True),
        ("Skewed",      4096,  4096, 1, torch.bfloat16, True,  False, True, True,  "swish",   0, 0, True),
    ]

    hdr = (
        f"{'Regime':<8} | {'Shape (M,GxN)':<15} | {'Variant':<21} | "
        f"{'LOP(us)':<9} | {'Tri(us)':<9} | {'Ref(us)':<10} | "
        f"{'vsTri':<7} | {'vsRef':<7} | {'HBM(GB/s)':<9} | {'%CopyBW':<7} | {'Toks/ms':<7}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows: list[dict[str, object]] = []
    for regime, m, n, groups, dt, is_rms, has_bias, has_z, nbg, act, pad_z, z_off, skewed in bench_cases:
        total_cols = groups * n
        x = torch.randn((m, total_cols), device="cuda", dtype=dt) * 0.5
        y = torch.empty_like(x)
        y_tri = torch.empty_like(x)
        w = torch.randn((total_cols,), device="cuda", dtype=dt) * 0.1 + 1.0
        b = torch.randn((total_cols,), device="cuda", dtype=dt) * 0.1 if has_bias else None
        z_full = (
            torch.randn((m, z_off + total_cols + pad_z), device="cuda", dtype=dt) * 0.5
            if has_z
            else None
        )
        z = z_full[:, z_off : z_off + total_cols] if z_full is not None else None
        if skewed:
            x[0::32].zero_()
            x[1::32] *= 50.0
            if z is not None:
                z[2::32].fill_(-40.0)
                z[3::32].fill_(40.0)
        mean = None if is_rms else torch.empty((groups, m), device="cuda", dtype=torch.float32)
        rstd = torch.empty((groups, m), device="cuda", dtype=torch.float32)

        # Pre-check accuracy before timing
        invoke_ln1p(
            layer_norm_fwd_1pass_opt, x, y, w, b, z, mean, rstd,
            m=m, n=n, eps=eps, has_bias=has_bias, has_z=has_z,
            norm_before_gate=nbg, is_rms_norm=is_rms, activation=act,
        )
        torch.cuda.synchronize()
        assert_matches_ref(
            x, y, w, b, z, mean, rstd,
            m=m, n=n, groups=groups, eps=eps,
            has_bias=has_bias, has_z=has_z, norm_before_gate=nbg,
            is_rms_norm=is_rms, activation=act,
            tag=f"bench_{regime}_{m}x{groups}x{n}",
        )

        iters = args.iters if (m * total_cols <= 4096 * 4096) else max(40, args.iters // 4)
        lop_us = measure_call_us(
            lambda: invoke_ln1p(
                layer_norm_fwd_1pass_opt, x, y, w, b, z, mean, rstd,
                m=m, n=n, eps=eps, has_bias=has_bias, has_z=has_z,
                norm_before_gate=nbg, is_rms_norm=is_rms, activation=act,
            ),
            args.warmup,
            iters,
        )
        tri_us: Optional[float] = None
        if triton_fn is not None:
            try:
                tri_us = measure_call_us(
                    lambda: triton_fn(
                        x, y_tri, w, b, z, mean, rstd,
                        m=m, n=n, groups=groups, eps=eps,
                        has_bias=has_bias, has_z=has_z, norm_before_gate=nbg,
                        is_rms_norm=is_rms, activation=act,
                    ),
                    args.warmup,
                    iters,
                )
            except Exception as exc:
                print(f"[warn] Triton failed on ({m},{groups}x{n}): {exc}")
                tri_us = None

        ref_us = measure_call_us(
            lambda: ref_layer_norm_fwd_1pass(
                x, w, b, z,
                m=m, n=n, groups=groups, eps=eps,
                has_bias=has_bias, has_z=has_z, norm_before_gate=nbg,
                is_rms_norm=is_rms, activation=act,
            ),
            args.warmup,
            iters,
        )

        es = x.element_size()
        # Steady-state HBM traffic (W/B read once from L2; X/Z read & Y/stats written).
        hbm_bytes = (
            m * total_cols * es * (3 if has_z else 2)
            + total_cols * es * (2 if has_bias else 1)
            + m * groups * 4 * (1 if is_rms else 2)
        )
        # Logical bytes (matching test_layer_norm_fwd_1pass.py convention counting W/B per row).
        logical_bytes = (
            m * total_cols * es * (3 + int(has_bias) + int(has_z))
            + m * groups * 4 * (1 if is_rms else 2)
        )
        hbm_bw = hbm_bytes / (lop_us * 1e-6) / 1e9
        logical_bw = logical_bytes / (lop_us * 1e-6) / 1e9
        pct_bw = (100.0 * hbm_bw / copy_bw_gbs) if copy_bw_gbs else None
        toks_per_ms = (m / (lop_us * 1e-3)) / 1000.0
        pct_str = f"{pct_bw:5.1f}%" if pct_bw is not None else "  N/A"
        vs_tri_str = f"{tri_us / lop_us:.2f}x" if tri_us else "N/A"
        tri_str = f"{tri_us:.2f}" if tri_us else "N/A"
        shape_str = f"{m}x{n}" if groups == 1 else f"{m}x{groups}x{n}"
        dt_tag = str(dt).removeprefix("torch.")[:4]
        var_str = (
            f"{dt_tag}/{'rms' if is_rms else 'ln'}/"
            f"{'post' if nbg else 'pre'}"
            f"{f'/p{pad_z}' if pad_z else ''}"
            f"{f'/o{z_off}' if z_off else ''}"
            f"{'/skw' if skewed else ''}"
        )
        print(
            f"{regime:<8} | {shape_str:<15} | {var_str:<21} | "
            f"{lop_us:<9.2f} | {tri_str:<9} | {ref_us:<10.2f} | "
            f"{vs_tri_str:<7} | {f'{ref_us / lop_us:.2f}x':<7} | "
            f"{hbm_bw:<9.2f} | {pct_str:<7} | {toks_per_ms:<7.1f}"
        )
        rows.append(
            {
                "op": "layer_norm_fwd_1pass_opt",
                "regime": regime,
                "M": m,
                "N": n,
                "groups": groups,
                "dtype": str(dt).removeprefix("torch."),
                "variant": var_str,
                "is_rms_norm": is_rms,
                "has_bias": has_bias,
                "norm_before_gate": nbg,
                "activation": act,
                "pad_z": pad_z,
                "z_offset": z_off,
                "skewed": skewed,
                "hbm_bytes": hbm_bytes,
                "logical_bytes": logical_bytes,
                "lightop_us": round(lop_us, 3),
                "triton_us": round(tri_us, 3) if tri_us is not None else "",
                "torch_ref_us": round(ref_us, 3),
                "speedup_vs_triton": round(tri_us / lop_us, 3) if tri_us is not None else "",
                "speedup_vs_ref": round(ref_us / lop_us, 3),
                "hbm_bw_gbs": round(hbm_bw, 2),
                "logical_bw_gbs": round(logical_bw, 2),
                "pct_copy_bw": round(pct_bw, 2) if pct_bw is not None else "",
                "toks_per_ms": round(toks_per_ms, 2),
            }
        )

    if args.csv and rows:
        csv_path = Path(args.csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[bench] CSV saved -> {csv_path}")


# =============================================================================
# Offline ISA Parser Self-Test & Pytest Entrypoints
# =============================================================================

def _build_synthetic_valid_gfx946_dump() -> str:
    """Minimal multi-arch dccobjdump fixture for offline parser verification."""
    return "\n".join(
        [
            '.amdgcn_target "amdgcn-amd-amdhsa--gfx938"',
            "_ZN2at6native39layer_norm_fwd_1pass_small_head_vec_kernelIN3c108BFloat16EfLi8ELi16ELi64ELb0ELb1ELb1ELb1ELNS0_14GateActivationE1EEEvPKT_PS4_S7_S7_S7_PfS8_lllllT0_S9_i:",
            "  .amdhsa_next_free_vgpr 72",
            "  .amdhsa_private_segment_fixed_size 0",
            "  ds_bpermute_b32 v1, v2, v3",
            "  v_div_scale_f32 v4, vcc, v5, v6, v5",
            "",
            '.amdgcn_target "amdgcn-amd-amdhsa--gfx946"',
            "_ZN2at6native39layer_norm_fwd_1pass_small_head_vec_kernelIN3c108BFloat16EfLi8ELi16ELi64ELb0ELb1ELb1ELb1ELNS0_14GateActivationE1EEEvPKT_PS4_S7_S7_S7_PfS8_lllllT0_S9_i:",
            "  .amdhsa_next_free_vgpr 48",
            "  .amdhsa_private_segment_fixed_size 0",
            "  v_dot2_f32_bf16 v0, v1, v1, v0",
            "  v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "  v_permlane16_swap_b32 v0, v3",
            "  v_permlane32_swap_b32 v0, v4",
            "  v_pk_mul_f32 v[6:7], v[8:9], v[10:11]",
            "  v_pk_add_f32 v[6:7], v[6:7], v[12:13]",
            "  v_pk_fma_f32 v[6:7], v[8:9], v[10:11], v[12:13]",
            "  v_exp_f32 v14, v15",
            "  v_rcp_f32 v16, v17",
            "  v_cvt_pk_bf16_f32 v18, v6, v7",
            "",
            "_ZN2at6native31layer_norm_fwd_1pass_vec_kernelIN3c104HalfEfLi8ELi128ELb0ELb1ELb1ELb1ELNS0_14GateActivationE1EEEvPKT_PS4_S7_S7_S7_PfS8_lllllT0_S9_i:",
            "  .amdhsa_next_free_vgpr 52",
            "  .amdhsa_private_segment_fixed_size 0",
            "  v_dot2_f32_f16 v0, v1, v1, v0",
            "  v_permlane16_swap_b32 v0, v2",
            "  v_permlane32_swap_b32 v0, v3",
            "  v_cvt_pk_f16_f32 v4, v5, v6",
            "",
            "_ZN2at6native37layer_norm_fwd_1pass_vec_multi_kernelIN3c108BFloat16EfLi8ELi256ELi2ELb0ELb1ELb0ELb1ELNS0_14GateActivationE1EEEvPKT_PS4_S7_S7_S7_PfS8_lllllT0_S9_i:",
            "  .amdhsa_next_free_vgpr 60",
            "  .amdhsa_private_segment_fixed_size 0",
            "  v_pk_fma_f32 v[0:1], v[2:3], v[2:3], v[0:1]",
            "  v_exp_f32 v4, v5",
            "  v_rcp_f32 v6, v7",
            "  v_cvt_pk_bf16_f32 v8, v0, v1",
        ]
    )


def test_layer_norm_fwd_1pass_gfx946_isa_parser_self_test() -> None:
    """Offline self-test verifying all 7 red-line checks in the ISA auditor."""
    valid_dump = _build_synthetic_valid_gfx946_dump()
    report = audit_dccobjdump_layer_norm_1pass_isa(valid_dump, strict=True)
    assert "SCRATCH=0" in report
    assert "raw_rms_fast maxVGPR=48 [<=64 target OK]" in report

    # Each red-line mutation must raise AssertionError.
    mutations = [
        ("scratch>0", valid_dump.replace(".amdhsa_private_segment_fixed_size 0", ".amdhsa_private_segment_fixed_size 16", 2)),
        ("same-reg permlane", valid_dump.replace("v_permlane16_swap_b32 v0, v3", "v_permlane16_swap_b32 v0, v0")),
        ("RTZ convert", valid_dump.replace("v_cvt_pk_f16_f32 v4, v5, v6", "v_cvt_pkrtz_f16_f32 v4, v5, v6")),
        ("IEEE div in vec", valid_dump.replace("v_rcp_f32 v16, v17", "v_div_scale_f32 v16, vcc, v17, v18, v17")),
        ("dot2 in pre-gate", valid_dump.replace("v_pk_fma_f32 v[0:1], v[2:3], v[2:3], v[0:1]", "v_dot2_f32_bf16 v0, v2, v2, v0")),
        ("FP64 in BF16", valid_dump.replace("v_pk_add_f32 v[6:7], v[6:7], v[12:13]", "v_add_f64 v[6:7], v[6:7], v[12:13]")),
        ("bpermute in vec", valid_dump.replace("v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1", "ds_bpermute_b32 v2, v0, v1")),
    ]
    for label, mutated in mutations:
        failed = False
        try:
            audit_dccobjdump_layer_norm_1pass_isa(mutated, strict=False)
        except AssertionError:
            failed = True
        assert failed, f"ISA auditor failed to catch red-line violation: {label}"


def test_layer_norm_fwd_1pass_gfx946_isa_audit() -> None:
    import pytest

    if not _ensure_torch(optional=True) or not torch.cuda.is_available():
        pytest.skip("需要 PyTorch HIP 与 CUDA/ROCm 设备")
    maybe_add_local_build_to_syspath()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = props.gcnArchName.split(":", 1)[0]
    ns = argparse.Namespace(
        so_path=None,
        isa_dump=None,
        require_gfx946=False,
        strict_isa=(arch == "gfx946"),
        skip_isa=False,
        isa_only=False,
    )
    stage_env_and_isa_check(ns)


def test_layer_norm_fwd_1pass_gfx946_correctness() -> None:
    import pytest

    if not _ensure_torch(optional=True) or not torch.cuda.is_available():
        pytest.skip("需要 PyTorch HIP 与 CUDA/ROCm 设备")
    maybe_add_local_build_to_syspath()
    ns = argparse.Namespace(
        quick=True,
        include_large_stress=False,
        stress_tokens=0,
        xcd_loops=3,
    )
    stage_correctness_and_edge_cases(ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified gfx946 verification, ISA audit & benchmark for layer_norm_fwd_1pass_opt"
    )
    parser.add_argument("--so-path", type=str, default=None, help="Explicit path to compiled lightop op*.so")
    parser.add_argument("--isa-dump", type=str, default=None, help="Optional saved dccobjdump text file for offline ISA audit")
    parser.add_argument("--require-gfx946", action="store_true", help="Fail if active GPU is not gfx946")
    parser.add_argument("--strict-isa", action="store_true", help="Fail if dccobjdump or expected gfx946 ISA hits are missing")
    parser.add_argument("--skip-isa", action="store_true", help="Skip dccobjdump / readelf ISA inspection")
    parser.add_argument("--isa-only", action="store_true", help="Only run Stage 1 environment and ISA inspection")
    parser.add_argument("--skip-bench", action="store_true", help="Skip performance benchmark stage")
    parser.add_argument("--skip-triton", action="store_true", help="Skip optional Triton baseline in Stage 3")
    parser.add_argument("--perfmodel", action="store_true", help="Preset for slow PerfModel (--quick + --skip-bench)")
    parser.add_argument("--quick", action="store_true", help="Smaller accuracy matrix and skip large stress tests")
    parser.add_argument("--include-large-stress", action="store_true", help="Run int64 stress even with --quick")
    parser.add_argument("--stress-tokens", type=int, default=32768, help="Rows for large-batch stress (default: 32768)")
    parser.add_argument("--xcd-loops", type=int, default=20, help="4-XCD consistency stress loop count (default: 20)")
    parser.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations (default: 20)")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark timed iterations (default: 200)")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path for benchmark results")
    args = parser.parse_args()
    if args.perfmodel:
        args.quick = True
        args.skip_bench = True
        args.xcd_loops = min(args.xcd_loops, 3)
    elif args.quick:
        args.xcd_loops = min(args.xcd_loops, 5)
    return args


def main() -> int:
    args = parse_args()
    offline_isa = bool(args.isa_only and (args.so_path or args.isa_dump))
    if _ensure_torch(optional=offline_isa):
        maybe_add_local_build_to_syspath()
    stage_env_and_isa_check(args)
    if args.isa_only:
        print("\nDONE (--isa-only).")
        return 0
    stage_correctness_and_edge_cases(args)
    if not args.skip_bench:
        stage_benchmark(args)
    print("\nALL GFX946 LAYER_NORM_FWD_1PASS VERIFICATION STAGES COMPLETED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
