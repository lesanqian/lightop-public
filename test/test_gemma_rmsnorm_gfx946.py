#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""gfx946 verification, ISA audit & benchmark suite for Gemma RMSNorm paths.

Covers all three Gemma RMSNorm paths (Route A arbitrary FP32 scale, zero API change,
aligned FP32 residual sum across residual paths):
  - Path 1 (No residual):  ``lightop.op.rms_norm_opt`` / ``lightop.gemma_rmsnorm``
  - Path 2 (Residual):     ``lightop.gemma_fused_add_rmsnorm`` / ``lightop.op.fused_add_rms_norm_opt``
  - Path 3 (FP8 fused):    ``lightop.gemma_rms_norm_per_token_fp8_quant`` / ``lightop.rms_norm_per_token_fp8_quant``

Stages (aligned with Group 1 & MoE routing gfx946 suites):
  1. Stage 1 — Environment & Binary ISA Audit (offline + online):
     - ``hy-smi`` / ``rocm-smi`` / ``rocminfo`` / ``torch`` device detection.
     - ``readelf --dyn-syms --wide`` check (zero unresolved ``__extendhfsf2``).
     - ``hipprof --codeobj-analyze`` excerpt when available.
     - ``dccobjdump`` disassembly audit on ``gfx946`` code objects in ``op*.so``:
       * Asserts ``SCRATCH == 0`` on all audited gfx946 RMSNorm kernels.
       * Asserts distinct VGPR operands on ``v_permlane{16,32}_swap_b32`` (errata R4).
       * Asserts zero ``v_cvt_pkrtz_`` (RTZ forbidden), zero ``v_cvt_scale_pk_{fp8,bf8}_``
         (Route A only), zero ``v_dot2_f32_bf16`` in residual kernels (preserves FP32 sum),
         and zero FP64 VALU inside BF16/FP16 kernel sections.
       * Tallies expected gfx946 instructions (``v_dot2_f32_bf16``, ``v_dot2_f32_f16``,
         ``v_pk_{add,mul,fma}_f32``, ``v_cvt_pk_{bf16,f16}_f32``, ``v_pk_max_{bf16,f16}``,
         ``v_permlane{16,32}_swap_b32``, ``v_cvt_pk_{fp8,bf8}_f32``) & peak VGPR count.
  2. Stage 2 — Full Boundary Accuracy, API Parity & 4-XCD Consistency Stress:
     - Zero-API-change signature & direct-binding parity checks.
     - Full dtype (BF16/FP16/FP32) x hidden-size x mode matrix covering small-hidden
       multi-token (64..512), fp32_tiled (4096/8192), vectorized opt (..16384), odd
       fallbacks (65/520/16392), clamp, in-place, unaligned offset=1, padded non-contiguous
       strides (pad=16 & pad=20 VMFault regression), 1D/3D, empty [0,H], zero rows,
       catastrophic cancellation (FP32 residual sum), RNE 0-ULP residual writeback, and
       HIP Graph capture/replay.
     - Large-batch ``int64`` indexing stress and repeated 4-XCD multi-block determinism
       stress (mixing zero, tiny, cancellation, and outlier rows across 4096+ tokens).
  3. Stage 3 — Decode / Prefill / Skewed Workload Benchmark + CSV Export:
     - Calibrates D2D HBM copy ceiling (``%CopyBW``) and compares LightOp vs PyTorch
       eager FP32 baselines across Decode, Prefill, and Skewed/Non-contiguous workloads.
     - Supplementary repo-style allclose/error metrics (max_abs/rmse/cosine/
       allclose@1e-2/1e-3/1e-4) mirroring test_rmsnorm.py and
       test_gemma_fused_add_rmsnorm.py are printed in Stage 3 (informational;
       the strict ULP/FP32 assertions above still decide PASS/FAIL).
     - Optional ``--baseline vllm`` adds the vLLM RMSNorm (+ triton FP8 for Path 3)
       unfused baseline and its relative speedup, mirroring the repository's
       test_gemma_rms_norm_fp8_quant.py. Default baseline stays pure-PyTorch FP32,
       so zero-dependency offline ISA audits are unaffected.

Usage (from lightop repo root):
  python3 test/test_gemma_rmsnorm_gfx946.py
  python3 test/test_gemma_rmsnorm_gfx946.py --isa-only --strict-isa --so-path path/to/op.so
  python3 test/test_gemma_rmsnorm_gfx946.py --perfmodel --require-gfx946 --strict-isa
  python3 test/test_gemma_rmsnorm_gfx946.py --require-gfx946 --strict-isa --csv gemma_rmsnorm_946.csv
  # Optional vLLM/triton unfused baseline + relative speedup (needs vllm/triton):
  python3 test/test_gemma_rmsnorm_gfx946.py --baseline vllm --csv gemma_rmsnorm_946.csv
  pytest -v test/test_gemma_rmsnorm_gfx946.py
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Optional

torch: Any = None
# Enabled by Stage 3 so the repo-style allclose metrics are printed only for
# benchmark cases (keeps the 600+ correctness cases quiet and ISA-only offline).
_VERBOSE_METRICS = False


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
    """Prefer repo-local ``build/lib.*`` or ``--inplace`` ``python/lightop`` if present."""
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

_RMS_KERNEL_RE = re.compile(
    r"(?:rms_kernel_small_hidden_multi_token"
    r"|fused_add_rms_kernel_small_hidden_multi_token"
    r"|fused_add_rms_kernel_fp32_tiled"
    r"|fused_add_rms_kernel_opt"
    r"|fused_rms_kernel_opt"
    r"|rms_norm_fp8_quant_residual_loop"
    r"|rms_norm_fp8_quant_residual"
    r"|rms_norm_fp8_quant_loop"
    r"|rms_norm_fp8_quant_)"
)

_KERNEL_SPLIT_RE = re.compile(
    r"(?=^[^\n]*(?:rms_kernel_small_hidden_multi_token"
    r"|fused_add_rms_kernel_small_hidden_multi_token"
    r"|fused_add_rms_kernel_fp32_tiled"
    r"|fused_add_rms_kernel_opt"
    r"|fused_rms_kernel_opt"
    r"|rms_norm_fp8_quant_)[^\n]*:|\t\.protected\t)",
    flags=re.MULTILINE,
)

_GFX946_SIGNAL_INSNS = (
    "v_dot2_f32_bf16",
    "v_pk_max_bf16",
    "v_permlane16_swap_b32",
    "v_permlane32_swap_b32",
    "v_pk_fma_f32",
    "v_pk_add_f32",
    "v_pk_mul_f32",
    "gfx946",
)


def _extract_arch_segments(dump_text: str) -> list[tuple[str, str]]:
    """Split multi-arch dccobjdump text into ``(detected_arch_or_unknown, text)`` sections."""
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


def _classify_rmsnorm_kernel(header: str) -> Optional[str]:
    if "rms_norm_fp8_quant_residual" in header:
        return "path3_fp8_residual"
    if "rms_norm_fp8_quant_" in header:
        return "path3_fp8_nores"
    if (
        "fused_add_rms_kernel_small_hidden_multi_token" in header
        or "fused_add_rms_kernel_fp32_tiled" in header
        or "fused_add_rms_kernel_opt" in header
    ):
        return "path2_residual"
    if (
        "rms_kernel_small_hidden_multi_token" in header
        or "fused_rms_kernel_opt" in header
    ):
        return "path1_nores"
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


def audit_dccobjdump_rmsnorm_isa(dump_text: str, strict: bool = False) -> str:
    """Parse dccobjdump output for gfx946 Gemma/RMSNorm kernels & verify ISA/resources."""
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
            head = blk[:420]
            if not _RMS_KERNEL_RE.search(head):
                continue
            fam = _classify_rmsnorm_kernel(head)
            if fam is not None:
                candidates.append((seg_arch, fam, blk))

    if not candidates:
        msg = "No Gemma/RMSNorm kernel blocks found in dccobjdump output"
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
        "v_pk_max_bf16": 0,
        "v_pk_max_f16": 0,
        "v_permlane16_swap_b32": 0,
        "v_permlane32_swap_b32": 0,
        "v_cvt_pk_fp8_f32": 0,
        "v_cvt_pk_bf8_f32": 0,
        "ds_bpermute_b32": 0,
    }
    audited_by_family = {
        "path1_nores": 0,
        "path2_residual": 0,
        "path3_fp8_nores": 0,
        "path3_fp8_residual": 0,
    }
    max_vgpr_by_family: dict[str, int] = {k: 0 for k in audited_by_family}

    scratch_violations: list[str] = []
    same_reg_swaps: list[str] = []
    forbidden_rtz: list[str] = []
    forbidden_scale_cast: list[str] = []
    forbidden_res_dot2: list[str] = []
    forbidden_fp64_in_16b: list[str] = []

    for seg_arch, fam, blk in candidates:
        has_946_signal = (seg_arch == "gfx946") or any(kw in blk for kw in _GFX946_SIGNAL_INSNS)
        if not has_946_signal:
            continue

        audited_by_family[fam] += 1
        header = blk.splitlines()[0].strip()[:120]

        scratch = _extract_block_scratch(blk)
        if scratch is not None and scratch != 0:
            scratch_violations.append(f"{header} (scratch={scratch})")

        vgpr = _extract_block_vgpr(blk)
        if vgpr is not None and vgpr > max_vgpr_by_family[fam]:
            max_vgpr_by_family[fam] = vgpr

        for kw in expected_insns:
            expected_insns[kw] += blk.count(kw)

        # Errata R4: distinct VGPRs required for permlane swap
        for swap_kw in ("v_permlane16_swap_b32", "v_permlane32_swap_b32"):
            for ln in blk.splitlines():
                if swap_kw in ln:
                    regs = re.findall(r"\bv\d+\b", ln)
                    if len(regs) >= 2 and regs[0] == regs[1]:
                        same_reg_swaps.append(f"{header}: {ln.strip()}")

        # Forbidden RTZ packed convert (violates RNE 0-ULP residual contract)
        if "v_cvt_pkrtz_" in blk:
            forbidden_rtz.append(f"{header} (v_cvt_pkrtz x{blk.count('v_cvt_pkrtz_')})")

        # Route A forbids E8M0-only scale cast instructions
        if "v_cvt_scale_pk_fp8" in blk or "v_cvt_scale_pk_bf8" in blk:
            forbidden_scale_cast.append(header)

        # Residual paths must keep FP32 residual sum — must NEVER use v_dot2_f32_bf16
        if fam in ("path2_residual", "path3_fp8_residual") and "v_dot2_f32_bf16" in blk:
            forbidden_res_dot2.append(header)

        # BF16/FP16 instantiations must not silently promote to 1/16-rate FP64 VALU
        if _is_half_or_bf16_symbol(header):
            for f64_kw in ("v_add_f64", "v_mul_f64", "v_fma_f64", "v_max_f64"):
                if f64_kw in blk:
                    forbidden_fp64_in_16b.append(f"{header} ({f64_kw})")
                    break

    total_audited = sum(audited_by_family.values())
    if total_audited == 0:
        msg = (
            "Found RMSNorm kernel blocks in dccobjdump, but none matched gfx946 "
            "signals (was op*.so built with gfx946?)"
        )
        if strict:
            raise AssertionError(msg)
        return f"[isa][warn] {msg}"

    if scratch_violations:
        raise AssertionError("Non-zero scratch spill in RMSNorm kernels: " + "; ".join(scratch_violations))
    if same_reg_swaps:
        raise AssertionError("Same-register operand in permlane swap: " + "; ".join(same_reg_swaps))
    if forbidden_rtz:
        raise AssertionError("Forbidden RTZ v_cvt_pkrtz_ in gfx946 RMSNorm kernels: " + "; ".join(forbidden_rtz))
    if forbidden_scale_cast:
        raise AssertionError("Unexpected Route-B v_cvt_scale_pk_* in Route-A kernels: " + "; ".join(forbidden_scale_cast))
    if forbidden_res_dot2:
        raise AssertionError(
            "v_dot2_f32_bf16 emitted in residual kernel (breaks FP32 residual sum): "
            + "; ".join(forbidden_res_dot2)
        )
    if forbidden_fp64_in_16b:
        raise AssertionError("Unexpected FP64 VALU in BF16/FP16 kernels: " + "; ".join(forbidden_fp64_in_16b))

    required_insns = (
        "v_dot2_f32_bf16",
        "v_pk_add_f32",
        "v_pk_fma_f32",
        "v_pk_mul_f32",
        "v_cvt_pk_bf16_f32",
        "v_cvt_pk_f16_f32",
        "v_pk_max_bf16",
        "v_pk_max_f16",
        "v_permlane16_swap_b32",
        "v_permlane32_swap_b32",
        "v_cvt_pk_fp8_f32",
        "v_cvt_pk_bf8_f32",
    )
    missing = [k for k in required_insns if expected_insns[k] == 0]
    missing_fams = [k for k, v in audited_by_family.items() if v == 0]
    if strict and (missing or missing_fams):
        raise AssertionError(
            f"Missing expected gfx946 ISA hits={missing} or kernel families={missing_fams}"
        )

    lines = [
        f"[isa] Audited {total_audited} gfx946 RMSNorm kernel section(s) "
        "(SCRATCH=0, distinct permlane regs, zero RTZ/FP64/Route-B/res-dot2):"
    ]
    for fam, cnt in audited_by_family.items():
        vgpr_str = f"maxVGPR={max_vgpr_by_family[fam]}" if max_vgpr_by_family[fam] > 0 else "maxVGPR=N/A"
        lines.append(f"  family {fam:<20}: {cnt:>3} section(s) ({vgpr_str})")
    for kw, cnt in expected_insns.items():
        lines.append(f"  {kw:<26}: {cnt} hit(s)")
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
    elif not (args.isa_only and args.so_path):
        raise RuntimeError("No CUDA/ROCm device (or PyTorch) available.")

    so_path = args.so_path
    if not so_path:
        import lightop
        from lightop import op as lightop_native

        so_path = getattr(lightop_native, "__file__", None)
        print(f"[lightop] pkg={lightop.__file__}")
    info["op_so"] = so_path
    print(f"[lightop] op_so={so_path}")

    if args.skip_isa:
        print("[isa] Skipped by --skip-isa")
        return info

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
            rms_lines = [
                ln.strip()
                for ln in out_ca.splitlines()
                if any(k in ln for k in ("rms_norm", "fused_add_rms", "fused_rms_kernel", "gfx946"))
            ]
            if rms_lines:
                print("[hipprof --codeobj-analyze excerpt]\n  " + "\n  ".join(rms_lines[:24]))

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

    print(audit_dccobjdump_rmsnorm_isa(out_d, strict=args.strict_isa))
    return info


# =============================================================================
# Independent PyTorch FP32 References & Validation Helpers
# =============================================================================

def fp8_ordinal(t: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude FP8 bytes -> monotonic signed ordinal for +0.0/-0.0-safe ULP diff."""
    u = t.view(torch.uint8).to(torch.int16)
    mag = u & 0x7F
    neg = (u & 0x80) != 0
    return torch.where(neg, -mag, mag)


def ref_rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    gemma: bool = False,
    limit: float = -1.0,
) -> torch.Tensor:
    xf = x.float()
    wf = w.float() + (1.0 if gemma else 0.0)
    y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps) * wf
    if limit >= 0.0:
        y = y.clamp(-limit, limit)
    return y.to(x.dtype)


def ref_fused_add_rms_norm(
    x: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    gemma: bool = False,
    limit: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x.float() + r.float()
    wf = w.float() + (1.0 if gemma else 0.0)
    y = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + eps) * wf
    if limit >= 0.0:
        y = y.clamp(-limit, limit)
    return y.to(x.dtype), summed.to(r.dtype)


def ref_rms_norm_per_token_fp8_quant(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    fp8type: int = 0,
    residual: Optional[torch.Tensor] = None,
    gemma: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    in_dtype = x.dtype
    fp8_dtype = torch.float8_e4m3fn if fp8type == 0 else torch.float8_e5m2
    fp8_max = 448.0 if fp8type == 0 else 57344.0
    min_scale = 1.0 / (fp8_max * 512.0)

    if residual is not None:
        summed = x.float() + residual.float()
        ref_res = summed.to(in_dtype)
        norm_in = summed
    else:
        ref_res = None
        norm_in = x.float()

    rstd = torch.rsqrt(norm_in.square().mean(dim=-1, keepdim=True) + eps)
    wt = w.float() + (1.0 if gemma else 0.0)
    elem = (norm_in * rstd * wt).to(in_dtype)

    absmax = elem.float().abs().amax(dim=-1, keepdim=True)
    scales = (absmax / fp8_max).clamp(min=min_scale)
    q = (elem.float() / scales).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q, scales.reshape(-1, 1), elem, ref_res


def _half_ulp_ordinal(t: torch.Tensor) -> torch.Tensor:
    """Signed-magnitude ordinal in 16-bit float bit space (adjacent values differ by 1).
    Works for both bf16 and fp16 (both are sign-magnitude 16-bit formats)."""
    u = t.view(torch.uint16).to(torch.int32)
    mag = u & 0x7FFF
    neg = (u & 0x8000) != 0
    return torch.where(neg, -mag, mag)


def _assert_half_within_ulps(actual: torch.Tensor, expected: torch.Tensor, tag: str, max_ulps: int = 1) -> None:
    """Half-precision RMSNorm may differ from the correctly-rounded FP32->half
    reference by an RNE tie-break of one ULP on odd/fallback hidden sizes.
    Verify elementwise agreement within ``max_ulps`` ULPs, which is the
    meaningful accuracy contract for half precision.
    """
    a = _half_ulp_ordinal(actual)
    e = _half_ulp_ordinal(expected)
    d = (a - e).abs()
    bad = int((d > max_ulps).sum().item())
    assert bad == 0, f"[{tag}] {bad} elements exceed {max_ulps} half-ULP " \
                     f"(max_ulp={int(d.max().item())})"


def _print_error_metrics(actual: torch.Tensor, expected: torch.Tensor, tag: str) -> None:
    """Print repo-style allclose/error metrics (informational only).

    Mirrors ``calc_metrics`` in test_gemma_fused_add_rmsnorm.py and the
    ``allclose`` checks in test_rmsnorm.py. The strict ULP / FP32 assertions in
    ``assert_path{1,2,3}_matches_ref`` still decide PASS/FAIL.
    """
    if not _VERBOSE_METRICS or actual.numel() == 0:
        return
    a = actual.float().detach().cpu()
    e = expected.float().detach().cpu()
    diff = a - e
    max_abs = diff.abs().max().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    cos = torch.nn.functional.cosine_similarity(
        a.reshape(-1).unsqueeze(0), e.reshape(-1).unsqueeze(0), dim=1
    ).item()
    ac12 = torch.allclose(a, e, atol=1e-2, rtol=1e-2)
    ac13 = torch.allclose(a, e, atol=1e-3, rtol=1e-3)
    ac14 = torch.allclose(a, e, atol=1e-4, rtol=1e-4)
    print(f"[metrics][{tag}] max_abs={max_abs:.3e} rmse={rmse:.3e} "
          f"cosine={cos:.6f} allclose@1e-2={ac12} @1e-3={ac13} @1e-4={ac14}")


def assert_path1_matches_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float,
    gemma: bool,
    limit: float,
    tag: str,
) -> float:
    expected = ref_rms_norm(x, w, eps, gemma=gemma, limit=limit)
    assert torch.isfinite(out).all(), f"[{tag}] non-finite output"
    rel_err = (
        (out.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-30)
    ).item()
    _print_error_metrics(out, expected, tag)
    if x.dtype == torch.float32:
        tol = 1e-5
        assert rel_err < tol, f"[{tag}] rel_err={rel_err:.6e} >= {tol}"
    elif x.dtype in (torch.bfloat16, torch.float16):
        _assert_half_within_ulps(out, expected, tag, max_ulps=2)
        assert rel_err < 1e-2, f"[{tag}] rel_err={rel_err:.6e} >= 1e-2"
    if limit >= 0.0 and out.numel() > 0:
        assert out.float().abs().max().item() <= limit + 1e-5, f"[{tag}] clamp violated"
    return rel_err


def assert_path2_matches_ref(
    x0: torch.Tensor,
    r0: torch.Tensor,
    w: torch.Tensor,
    x_act: torch.Tensor,
    r_act: torch.Tensor,
    *,
    eps: float,
    gemma: bool,
    limit: float,
    tag: str,
) -> float:
    exp_x, exp_r = ref_fused_add_rms_norm(x0, r0, w, eps, gemma=gemma, limit=limit)
    assert torch.isfinite(x_act).all(), f"[{tag}] non-finite output"
    assert torch.isfinite(r_act).all(), f"[{tag}] non-finite residual"
    torch.testing.assert_close(r_act, exp_r, rtol=0, atol=0, msg=f"[{tag}] residual RNE mismatch")
    rel_err = (
        (x_act.float() - exp_x.float()).norm()
        / exp_x.float().norm().clamp_min(1e-30)
    ).item()
    _print_error_metrics(x_act, exp_x, tag)
    if x0.dtype == torch.float32:
        tol = 1e-5
        assert rel_err < tol, f"[{tag}] rel_err={rel_err:.6e} >= {tol}"
    elif x0.dtype in (torch.bfloat16, torch.float16):
        _assert_half_within_ulps(x_act, exp_x, tag, max_ulps=2)
        assert rel_err < 1e-2, f"[{tag}] rel_err={rel_err:.6e} >= 1e-2"
    if limit >= 0.0 and x_act.numel() > 0:
        assert x_act.float().abs().max().item() <= limit + 1e-5, f"[{tag}] clamp violated"
    return rel_err


def assert_path3_matches_ref(
    x0: torch.Tensor,
    r0: Optional[torch.Tensor],
    w0: torch.Tensor,
    x_after: torch.Tensor,
    r_after: Optional[torch.Tensor],
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    *,
    eps: float,
    fp8type: int,
    gemma: bool,
    update_input: bool,
    tag: str,
    zero_case: bool = False,
) -> float:
    ref_q, ref_s, ref_elem, ref_res = ref_rms_norm_per_token_fp8_quant(
        x0, w0, eps, fp8type=fp8type, residual=r0, gemma=gemma
    )
    assert torch.isfinite(out_s).all(), f"[{tag}] non-finite scales"
    assert torch.isfinite(out_q.float()).all(), f"[{tag}] non-finite FP8 output"

    if r0 is not None:
        assert r_after is not None
        torch.testing.assert_close(
            r_after, ref_res, rtol=0, atol=0, msg=f"[{tag}] residual RNE mismatch"
        )

    if update_input:
        elem_err = (
            (x_after.float() - ref_elem.float()).norm()
            / ref_elem.float().norm().clamp_min(1e-30)
        ).item()
        assert elem_err < 5e-3, f"[{tag}] updated input rel_err={elem_err:.6e}"
    else:
        torch.testing.assert_close(x_after, x0, rtol=0, atol=0, msg=f"[{tag}] input mutated when update_input=False")

    if zero_case:
        fp8_max = 448.0 if fp8type == 0 else 57344.0
        min_scale = 1.0 / (fp8_max * 512.0)
        torch.testing.assert_close(
            out_s, torch.full_like(out_s, min_scale), rtol=1e-6, atol=1e-12, msg=f"[{tag}] min_scale mismatch"
        )
        torch.testing.assert_close(
            out_q.float(), torch.zeros_like(out_q.float()), rtol=0, atol=0, msg=f"[{tag}] zero_case non-zero output"
        )
        return 0.0

    scale_rtol = 1e-2 if x0.dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out_s, ref_s, rtol=scale_rtol, atol=1e-6, msg=f"[{tag}] scale mismatch")

    deq_act = out_q.reshape(-1, x0.shape[-1]).float() * out_s
    deq_ref = ref_q.reshape(-1, x0.shape[-1]).float() * ref_s
    rel_err = ((deq_act - deq_ref).norm() / deq_ref.norm().clamp_min(1e-30)).item()
    _print_error_metrics(deq_act, deq_ref, tag)
    tol = 5e-2 if fp8type == 0 else 1.2e-1
    assert rel_err < tol, f"[{tag}] dequant rel_err={rel_err:.6e} >= {tol}"
    return rel_err


# =============================================================================
# Stage 2: Full Boundary Accuracy, API Parity & 4-XCD Consistency Stress
# =============================================================================

def stage_correctness_and_edge_cases(args: argparse.Namespace) -> None:
    print("\n================ 2. CORRECTNESS, EDGE CASES & 4-XCD STRESS ================")
    import lightop
    from lightop import op as lightop_native
    from lightop.norm import (
        gemma_fused_add_rmsnorm,
        gemma_rms_norm_per_token_fp8_quant,
        gemma_rmsnorm,
        rms_norm_opt,
        rms_norm_per_token_fp8_quant,
    )

    eps = 1e-5

    # 2.1 Zero-API-change signature & categorized binding parity
    assert rms_norm_opt is lightop_native.rms_norm_opt
    assert list(inspect.signature(gemma_rmsnorm).parameters) == [
        "x", "weight", "variance_epsilon", "out",
    ]
    assert list(inspect.signature(gemma_fused_add_rmsnorm).parameters) == [
        "x", "residual", "weight", "variance_epsilon",
    ]
    assert list(inspect.signature(gemma_rms_norm_per_token_fp8_quant).parameters) == [
        "input", "weight", "epsilon", "fp8type", "residual", "update_input",
    ]
    assert lightop.gemma_fused_add_rmsnorm is gemma_fused_add_rmsnorm
    assert lightop.gemma_rms_norm_per_token_fp8_quant is gemma_rms_norm_per_token_fp8_quant
    print("  [PASS] Zero-API-change signatures & direct/categorized binding parity")

    torch.manual_seed(946)

    # 2.2 Path 1 (No residual): rms_norm_opt & gemma_rmsnorm
    p1_hiddens = (
        (64, 128, 256, 512, 1024, 3584, 4096, 8192, 65, 520)
        if args.quick
        else (64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3584, 4096, 5376, 8192, 16384, 65, 520, 16392)
    )
    p1_cases = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for h in p1_hiddens:
            x = torch.randn((17, h), device="cuda", dtype=dtype)
            w = torch.randn((h,), device="cuda", dtype=dtype)
            x0, w0 = x.clone(), w.clone()
            for gemma in (False, True):
                out = torch.empty_like(x)
                if gemma:
                    gemma_rmsnorm(x, w, eps, out=out)
                else:
                    lightop_native.rms_norm_opt(out, x, w, eps, -1.0)
                torch.cuda.synchronize()
                torch.testing.assert_close(x, x0, rtol=0, atol=0)
                torch.testing.assert_close(w, w0, rtol=0, atol=0)
                assert_path1_matches_ref(x0, w0, out, eps=eps, gemma=gemma, limit=-1.0, tag=f"p1_{dtype}_H{h}_g{gemma}")
                p1_cases += 1

    # Path 1 edge cases: clamp, in-place, unaligned out (offset=1), 3D, empty [0, H], grid=288
    for dtype in (torch.bfloat16, torch.float16):
        for h in (128, 512, 4096):
            # Clamp
            x = torch.randn((17, h), device="cuda", dtype=dtype)
            w = torch.randn((h,), device="cuda", dtype=dtype)
            out = torch.empty_like(x)
            lightop_native.rms_norm_opt(out, x, w, eps, 0.5)
            torch.cuda.synchronize()
            assert_path1_matches_ref(x, w, out, eps=eps, gemma=False, limit=0.5, tag=f"p1_clamp_{dtype}_H{h}")
            # In-place
            x_ip = x.clone()
            gemma_rmsnorm(x_ip, w, eps)
            torch.cuda.synchronize()
            assert_path1_matches_ref(x, w, x_ip, eps=eps, gemma=True, limit=-1.0, tag=f"p1_inplace_{dtype}_H{h}")
            # Unaligned output tensor (offset=1)
            out_un = torch.empty(17 * h + 1, device="cuda", dtype=dtype)[1:].view(17, h)
            lightop_native.gemma_rmsnorm(out_un, x, w, eps)
            torch.cuda.synchronize()
            assert_path1_matches_ref(x, w, out_un, eps=eps, gemma=True, limit=-1.0, tag=f"p1_unaligned_out_{dtype}_H{h}")
            # T=288 (128-CU vs 192-CU grid rule)
            x288 = torch.randn((288, h), device="cuda", dtype=dtype)
            o288 = torch.empty_like(x288)
            lightop_native.gemma_rmsnorm(o288, x288, w, eps)
            torch.cuda.synchronize()
            assert_path1_matches_ref(x288, w, o288, eps=eps, gemma=True, limit=-1.0, tag=f"p1_T288_{dtype}_H{h}")
            p1_cases += 4

    # 3D & empty
    x3d = torch.randn((2, 7, 128), device="cuda", dtype=torch.bfloat16)
    w3d = torch.randn((128,), device="cuda", dtype=torch.bfloat16)
    o3d = torch.empty_like(x3d)
    lightop_native.gemma_rmsnorm(o3d, x3d, w3d, eps)
    torch.cuda.synchronize()
    assert_path1_matches_ref(x3d, w3d, o3d, eps=eps, gemma=True, limit=-1.0, tag="p1_3d")
    for h in (65, 128, 512, 4096):
        xe = torch.empty((0, h), device="cuda", dtype=torch.bfloat16)
        oe = torch.empty_like(xe)
        we = torch.ones((h,), device="cuda", dtype=torch.bfloat16)
        lightop_native.rms_norm_opt(oe, xe, we, eps, -1.0)
        lightop_native.gemma_rmsnorm(oe, xe, we, eps)
    torch.cuda.synchronize()
    print(f"  [PASS] Path 1 (rms_norm_opt / gemma_rmsnorm) matrix & boundaries ({p1_cases} cases)")

    # 2.3 Path 2 (Residual, strict FP32 sum): gemma_fused_add_rmsnorm & fused_add_rms_norm_opt
    p2_cases = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for h in p1_hiddens:
            for gemma in (False, True):
                x = torch.randn((17, h), device="cuda", dtype=dtype)
                r = torch.randn_like(x)
                w = torch.randn((h,), device="cuda", dtype=dtype)
                x0, r0, w0 = x.clone(), r.clone(), w.clone()
                if gemma:
                    gemma_fused_add_rmsnorm(x, r, w, eps)
                else:
                    lightop_native.fused_add_rms_norm_opt(x, r, w, eps, -1.0)
                torch.cuda.synchronize()
                torch.testing.assert_close(w, w0, rtol=0, atol=0)
                assert_path2_matches_ref(x0, r0, w0, x, r, eps=eps, gemma=gemma, limit=-1.0, tag=f"p2_{dtype}_H{h}_g{gemma}")
                p2_cases += 1

    # Path 2 edges: catastrophic cancellation, unaligned offset=1, clamp, T=288, -0.0, empty, HIP Graph
    for dtype in (torch.bfloat16, torch.float16):
        for h in (65, 128, 512, 4096, 8192):
            # Cancellation (detects premature BF16/FP16 rounding or dot2_bf16 regression)
            x = torch.randn((17, h), device="cuda", dtype=dtype)
            r = (-x.float() + 0.003).to(dtype)
            w = torch.randn((h,), device="cuda", dtype=dtype)
            x0, r0 = x.clone(), r.clone()
            gemma_fused_add_rmsnorm(x, r, w, eps)
            torch.cuda.synchronize()
            assert_path2_matches_ref(x0, r0, w, x, r, eps=eps, gemma=True, limit=-1.0, tag=f"p2_cancel_{dtype}_H{h}")
            # Unaligned offset=1
            x_u = torch.randn(17 * h + 1, device="cuda", dtype=dtype)[1:].view(17, h)
            r_u = torch.randn(17 * h + 1, device="cuda", dtype=dtype)[1:].view(17, h)
            w_u = torch.randn(h + 1, device="cuda", dtype=dtype)[1:]
            xu0, ru0 = x_u.clone(), r_u.clone()
            gemma_fused_add_rmsnorm(x_u, r_u, w_u, eps)
            torch.cuda.synchronize()
            assert_path2_matches_ref(xu0, ru0, w_u, x_u, r_u, eps=eps, gemma=True, limit=-1.0, tag=f"p2_unaligned_{dtype}_H{h}")
            # Clamp + signed zero
            x_c = torch.randn((17, h), device="cuda", dtype=dtype)
            x_c[0, 0] = -0.0
            r_c = torch.randn_like(x_c)
            r_c[0, 0] = -0.0
            xc0, rc0 = x_c.clone(), r_c.clone()
            lightop_native.fused_add_rms_norm_opt(x_c, r_c, w, eps, 0.5)
            torch.cuda.synchronize()
            assert_path2_matches_ref(xc0, rc0, w, x_c, r_c, eps=eps, gemma=False, limit=0.5, tag=f"p2_clamp_{dtype}_H{h}")
            p2_cases += 3

        for h in (2048, 4096, 8192):
            x288 = torch.randn((288, h), device="cuda", dtype=dtype)
            r288 = torch.randn_like(x288)
            w288 = torch.randn((h,), device="cuda", dtype=dtype)
            x0, r0 = x288.clone(), r288.clone()
            gemma_fused_add_rmsnorm(x288, r288, w288, eps)
            torch.cuda.synchronize()
            assert_path2_matches_ref(x0, r0, w288, x288, r288, eps=eps, gemma=True, limit=-1.0, tag=f"p2_T288_{dtype}_H{h}")
            p2_cases += 1

    # Path 2 HIP Graph replay
    for h in (128, 4096, 8192):
        x = torch.randn((17, h), device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        w = torch.randn((h,), device="cuda", dtype=torch.bfloat16)
        x0, r0 = x.clone(), r.clone()
        gemma_fused_add_rmsnorm(x, r, w, eps)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            x.copy_(x0)
            r.copy_(r0)
            gemma_fused_add_rmsnorm(x, r, w, eps)
        for rep in range(3):
            x0.normal_()
            r0.normal_()
            g.replay()
            torch.cuda.synchronize()
            assert_path2_matches_ref(x0, r0, w, x, r, eps=eps, gemma=True, limit=-1.0, tag=f"p2_graph_H{h}_r{rep}")
    print(f"  [PASS] Path 2 (gemma_fused_add_rmsnorm / fp32_tiled) matrix, cancellation & HIP Graph ({p2_cases} cases)")

    # 2.4 Path 3 (FP8 Fused, Route A FP32 scale, aligned FP32 residual sum)
    p3_hiddens = (
        (64, 128, 512, 1024, 2048, 4096, 8192, 65, 520)
        if args.quick
        else (64, 128, 256, 512, 1024, 2048, 2560, 3584, 4096, 5376, 8192, 16384, 65, 520, 16392)
    )
    p3_cases = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for h in p3_hiddens:
            for fp8type in (0, 1):
                for with_res in (False, True):
                    for gemma in ((True,) if (args.quick and dtype == torch.float32) else (True, False)):
                        x = torch.randn((17, h), device="cuda", dtype=dtype)
                        r = torch.randn_like(x) if with_res else None
                        w = torch.randn((h,), device="cuda", dtype=dtype)
                        x0 = x.clone()
                        r0 = r.clone() if r is not None else None
                        w0 = w.clone()
                        upd = (h in (128, 1024, 4096, 520)) and gemma
                        fn = gemma_rms_norm_per_token_fp8_quant if gemma else rms_norm_per_token_fp8_quant
                        oq, os_ = fn(x, w, epsilon=eps, fp8type=fp8type, residual=r, update_input=upd)
                        torch.cuda.synchronize()
                        assert_path3_matches_ref(
                            x0, r0, w0, x, r, oq, os_,
                            eps=eps, fp8type=fp8type, gemma=gemma, update_input=upd,
                            tag=f"p3_{dtype}_H{h}_fp8={fp8type}_res={with_res}_g={gemma}_upd={upd}",
                        )
                        p3_cases += 1

    # Path 3 hardware edge cases: cancellation, all-zero, tiny/saturation, pad=16/20, unaligned, 1D/3D, T=140, empty, Graph
    for dtype in (torch.bfloat16, torch.float16):
        # (a) Residual FP32 cancellation
        for h in (128, 512, 1024, 4096, 8192, 520):
            x = torch.randn((17, h), device="cuda", dtype=dtype)
            r = (-x.float() + 0.003).to(dtype)
            w = torch.randn((h,), device="cuda", dtype=dtype)
            x0, r0 = x.clone(), r.clone()
            oq, os_ = gemma_rms_norm_per_token_fp8_quant(x, w, epsilon=eps, fp8type=0, residual=r, update_input=True)
            torch.cuda.synchronize()
            assert_path3_matches_ref(
                x0, r0, w, x, r, oq, os_,
                eps=eps, fp8type=0, gemma=True, update_input=True,
                tag=f"p3_cancel_{dtype}_H{h}",
            )
            p3_cases += 1

        # (b) All-zero rows -> min_scaling floor, no NaN/Inf
        for fp8type in (0, 1):
            for with_res in (False, True):
                z = torch.zeros((8, 1024), device="cuda", dtype=dtype)
                rz = torch.zeros_like(z) if with_res else None
                wz = torch.randn((1024,), device="cuda", dtype=dtype)
                z0 = z.clone()
                rz0 = rz.clone() if rz is not None else None
                oq, os_ = gemma_rms_norm_per_token_fp8_quant(z, wz, epsilon=eps, fp8type=fp8type, residual=rz)
                torch.cuda.synchronize()
                assert_path3_matches_ref(
                    z0, rz0, wz, z, rz, oq, os_,
                    eps=eps, fp8type=fp8type, gemma=True, update_input=False,
                    tag=f"p3_zero_{dtype}_fp8={fp8type}_res={with_res}", zero_case=True,
                )
                p3_cases += 1

        # (c) Non-contiguous padded strides: pad=16 (16B aligned fast path) & pad=20 (loop VMFault regression)
        for pad in (16, 20):
            for with_res in (False, True):
                x = torch.randn((17, 1024 + pad), device="cuda", dtype=dtype)[:, :1024]
                r = torch.randn((17, 1024 + pad), device="cuda", dtype=dtype)[:, :1024] if with_res else None
                w = torch.randn((1024,), device="cuda", dtype=dtype)
                x0 = x.clone()
                r0 = r.clone() if r is not None else None
                oq, os_ = gemma_rms_norm_per_token_fp8_quant(
                    x, w, epsilon=eps, fp8type=0, residual=r, update_input=True
                )
                torch.cuda.synchronize()
                assert_path3_matches_ref(
                    x0, r0, w, x, r, oq, os_,
                    eps=eps, fp8type=0, gemma=True, update_input=True,
                    tag=f"p3_pad{pad}_{dtype}_res={with_res}",
                )
                p3_cases += 1

        # (d) Unaligned offset=1 & tiny/outlier saturation
        for h in (512, 4096):
            x_u = torch.randn(17 * h + 1, device="cuda", dtype=dtype)[1:].view(17, h)
            x_u[0].fill_(1e-6)
            x_u[1, 0] = 500.0
            x_u[1, 1] = -500.0
            x_u[2, 0] = -0.0
            r_u = torch.randn(17 * h + 1, device="cuda", dtype=dtype)[1:].view(17, h)
            w_u = torch.randn(h + 1, device="cuda", dtype=dtype)[1:]
            xu0, ru0 = x_u.clone(), r_u.clone()
            oq, os_ = gemma_rms_norm_per_token_fp8_quant(x_u, w_u, epsilon=eps, fp8type=0, residual=r_u)
            torch.cuda.synchronize()
            assert_path3_matches_ref(
                xu0, ru0, w_u, x_u, r_u, oq, os_,
                eps=eps, fp8type=0, gemma=True, update_input=False,
                tag=f"p3_unaligned_sat_{dtype}_H{h}",
            )
            p3_cases += 1

    # (e) 1D, 3D, T=140 grid_opt boundary, empty [0, H], HIP Graph
    for shape in ((1024,), (2, 7, 4096), (140, 4096), (140, 8192)):
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        w = torch.randn((shape[-1],), device="cuda", dtype=torch.bfloat16)
        x0, r0 = x.clone(), r.clone()
        oq, os_ = gemma_rms_norm_per_token_fp8_quant(x, w, epsilon=eps, fp8type=0, residual=r)
        torch.cuda.synchronize()
        assert_path3_matches_ref(
            x0, r0, w, x, r, oq, os_,
            eps=eps, fp8type=0, gemma=True, update_input=False,
            tag=f"p3_shape_{shape}",
        )
        p3_cases += 1

    for h in (65, 512, 4096):
        xe = torch.empty((0, h), device="cuda", dtype=torch.bfloat16)
        re_ = torch.empty_like(xe)
        we = torch.ones((h,), device="cuda", dtype=torch.bfloat16)
        oq, os_ = gemma_rms_norm_per_token_fp8_quant(xe, we, epsilon=eps, fp8type=0, residual=re_)
        assert oq.shape == (0, h) and os_.shape == (0, 1)
    torch.cuda.synchronize()

    for h in (1024, 4096):
        for with_res in (False, True):
            x = torch.randn((17, h), device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x) if with_res else None
            w = torch.randn((h,), device="cuda", dtype=torch.bfloat16)
            x0 = x.clone()
            r0 = r.clone() if r is not None else None
            gemma_rms_norm_per_token_fp8_quant(x, w, epsilon=eps, fp8type=0, residual=r)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                x.copy_(x0)
                if r is not None:
                    r.copy_(r0)
                oq_g, os_g = gemma_rms_norm_per_token_fp8_quant(x, w, epsilon=eps, fp8type=0, residual=r)
            for rep in range(3):
                x0.normal_()
                if r0 is not None:
                    r0.normal_()
                g.replay()
                torch.cuda.synchronize()
                assert_path3_matches_ref(
                    x0, r0, w, x, r, oq_g, os_g,
                    eps=eps, fp8type=0, gemma=True, update_input=False,
                    tag=f"p3_graph_H{h}_res={with_res}_r{rep}",
                )
    print(f"  [PASS] Path 3 (gemma_rms_norm_per_token_fp8_quant) matrix, edges & HIP Graph ({p3_cases} cases)")

    # 2.5 Large-batch int64 indexing stress
    if (not args.quick or args.include_large_stress) and args.stress_tokens > 0:
        st = args.stress_tokens
        sh = 4096
        x_big = torch.randn((st, sh), device="cuda", dtype=torch.bfloat16)
        r_big = torch.randn_like(x_big)
        w_big = torch.randn((sh,), device="cuda", dtype=torch.bfloat16)
        o_big = torch.empty_like(x_big)

        # Path 1
        lightop_native.gemma_rmsnorm(o_big, x_big, w_big, eps)
        torch.cuda.synchronize()
        for sl in (slice(0, 16), slice(st - 16, st)):
            assert_path1_matches_ref(x_big[sl], w_big, o_big[sl], eps=eps, gemma=True, limit=-1.0, tag="stress_p1")

        # Path 3 (non-residual + residual on tail slice verification)
        oq_b, os_b = gemma_rms_norm_per_token_fp8_quant(x_big, w_big, epsilon=eps, fp8type=0, residual=None)
        torch.cuda.synchronize()
        for sl in (slice(0, 16), slice(st - 16, st)):
            assert_path3_matches_ref(
                x_big[sl], None, w_big, x_big[sl], None, oq_b[sl], os_b[sl],
                eps=eps, fp8type=0, gemma=True, update_input=False, tag="stress_p3_nores",
            )

        # Path 2 (in-place on head/tail snapshot)
        x_head0, r_head0 = x_big[:16].clone(), r_big[:16].clone()
        x_tail0, r_tail0 = x_big[-16:].clone(), r_big[-16:].clone()
        gemma_fused_add_rmsnorm(x_big, r_big, w_big, eps)
        torch.cuda.synchronize()
        assert_path2_matches_ref(x_head0, r_head0, w_big, x_big[:16], r_big[:16], eps=eps, gemma=True, limit=-1.0, tag="stress_p2_head")
        assert_path2_matches_ref(x_tail0, r_tail0, w_big, x_big[-16:], r_big[-16:], eps=eps, gemma=True, limit=-1.0, tag="stress_p2_tail")
        del x_big, r_big, o_big, oq_b, os_b
        print(f"  [PASS] Large-batch int64 stress across Paths 1/2/3 ({st}x{sh}, {(st * sh) / 1e6:.1f}M elems)")

    # 2.6 4-XCD multi-block determinism & skewed-distribution consistency stress
    xcd_tokens = 2048 if args.quick else 4096
    xcd_loops = args.xcd_loops
    for h in (128, 4096, 8192):
        # Mix normal, zero, tiny, outlier, and cancellation rows across all 4 XCDs (192 CUs)
        x_base = torch.randn((xcd_tokens, h), device="cuda", dtype=torch.bfloat16)
        r_base = torch.randn_like(x_base)
        w_base = torch.randn((h,), device="cuda", dtype=torch.bfloat16)
        x_base[0::64].zero_()
        r_base[0::64].zero_()
        x_base[1::64].fill_(1e-5)
        r_base[1::64].fill_(-1e-5)
        x_base[2::64] *= 50.0
        r_base[3::64].copy_((-x_base[3::64].float() + 0.003).to(torch.bfloat16))

        # Path 1 determinism
        p1_ref_out = torch.empty_like(x_base)
        lightop_native.gemma_rmsnorm(p1_ref_out, x_base, w_base, eps)
        torch.cuda.synchronize()
        p1_buf = torch.empty_like(x_base)
        for it in range(xcd_loops):
            lightop_native.gemma_rmsnorm(p1_buf, x_base, w_base, eps)
            torch.cuda.synchronize()
            torch.testing.assert_close(p1_buf, p1_ref_out, rtol=0, atol=0, msg=f"[4xcd p1 H={h} it={it}]")

        # Path 2 determinism + reference check on skewed slice
        p2_x0, p2_r0 = x_base.clone(), r_base.clone()
        gemma_fused_add_rmsnorm(p2_x0, p2_r0, w_base, eps)
        torch.cuda.synchronize()
        assert_path2_matches_ref(
            x_base[:128], r_base[:128], w_base, p2_x0[:128], p2_r0[:128],
            eps=eps, gemma=True, limit=-1.0, tag=f"4xcd_p2_ref_H{h}",
        )
        p2_x = torch.empty_like(x_base)
        p2_r = torch.empty_like(r_base)
        for it in range(xcd_loops):
            p2_x.copy_(x_base)
            p2_r.copy_(r_base)
            gemma_fused_add_rmsnorm(p2_x, p2_r, w_base, eps)
            torch.cuda.synchronize()
            torch.testing.assert_close(p2_x, p2_x0, rtol=0, atol=0, msg=f"[4xcd p2 x H={h} it={it}]")
            torch.testing.assert_close(p2_r, p2_r0, rtol=0, atol=0, msg=f"[4xcd p2 r H={h} it={it}]")

        # Path 3 determinism (with residual, E4M3) + reference check
        p3_r0 = r_base.clone()
        p3_q0, p3_s0 = gemma_rms_norm_per_token_fp8_quant(
            x_base, w_base, epsilon=eps, fp8type=0, residual=p3_r0, update_input=False
        )
        torch.cuda.synchronize()
        assert_path3_matches_ref(
            x_base[:128], r_base[:128], w_base, x_base[:128], p3_r0[:128], p3_q0[:128], p3_s0[:128],
            eps=eps, fp8type=0, gemma=True, update_input=False, tag=f"4xcd_p3_ref_H{h}",
        )
        p3_r = torch.empty_like(r_base)
        for it in range(xcd_loops):
            p3_r.copy_(r_base)
            p3_q, p3_s = gemma_rms_norm_per_token_fp8_quant(
                x_base, w_base, epsilon=eps, fp8type=0, residual=p3_r, update_input=False
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(p3_q.view(torch.uint8), p3_q0.view(torch.uint8), rtol=0, atol=0, msg=f"[4xcd p3 q H={h} it={it}]")
            torch.testing.assert_close(p3_s, p3_s0, rtol=0, atol=0, msg=f"[4xcd p3 s H={h} it={it}]")
            torch.testing.assert_close(p3_r, p3_r0, rtol=0, atol=0, msg=f"[4xcd p3 r H={h} it={it}]")

    print(f"  [PASS] 4-XCD multi-block determinism & skewed-row stress ({xcd_loops}x iters, T={xcd_tokens}, H=128/4096/8192)")


# =============================================================================
# Stage 3: Decode / Prefill / Skewed Workload Benchmark + CSV Export
# =============================================================================

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
        nbytes = 2 * elems * 4
        return nbytes / (us * 1e-6) / 1e9
    except RuntimeError:
        return None


_VLLM_SHIMS_DONE = False
_VLLM_LAYERS: dict = {}
_VLLM_QUANT_FN = None


def _ensure_vllm_baseline_shims() -> None:
    """Apply the same env shims the repo's fp8-quant cross-validation driver uses.

    Import is entirely lazy so the default (pure-PyTorch) path and the offline
    ``--isa-only --so-path`` audit remain zero-dependency.
    """
    global _VLLM_SHIMS_DONE
    if _VLLM_SHIMS_DONE:
        return
    import torch.library

    # (a) torchvision C++ extension ABI-mismatch -> nms/qnms schemas undefined.
    for _name in ("torchvision::nms", "torchvision::qnms"):
        try:
            torch.library.define(
                _name, "(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
            )
        except Exception:
            pass
    # (b) bare VllmConfig() has no model_config.architectures for HY4 DCP check.
    try:
        import vllm_hcu.models.hy_v4.dcp_config as _dcp
        _dcp.validate_hy4_dcp_config = lambda vllm_config: False
    except Exception:
        pass
    # (c) this image's DTK clang rejects/silently drops some hardcoded -mllvm flags.
    try:
        from triton.backends.hcu import compiler as _hcu_comp
        _bad = {
            "-mllvm=-check-valu-data-forward-hazards=0",
            "-mllvm=-disable-cluster-lds-memops=true",
            "-mllvm=-fp-fuse=fast",
            "-mllvm=-hcu-pre-emit-load-store-opt=false",
        }
        _orig = _hcu_comp.HIPBackend._get_clang_args

        def _patched(metadata, options):
            return [a for a in _orig(metadata, options) if a not in _bad]

        _hcu_comp.HIPBackend._get_clang_args = staticmethod(_patched)
    except Exception:
        pass
    _VLLM_SHIMS_DONE = True


def _get_vllm_layer(hidden: int, eps: float, dtype, w):
    """Return a vLLM RMSNorm layer for the given shape, with ``w`` copied in."""
    key = (hidden, eps, dtype)
    layer = _VLLM_LAYERS.get(key)
    if layer is None:
        _ensure_vllm_baseline_shims()
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.layernorm import RMSNorm

        with set_current_vllm_config(VllmConfig()):
            layer = RMSNorm(hidden, eps).to(dtype=dtype, device="cuda")
        _VLLM_LAYERS[key] = layer
    layer.weight.data.copy_(w.to(dtype))
    return layer


def _get_vllm_quant_fn():
    """Return the triton per-token FP8 quant callable from the repo's fp8 script."""
    global _VLLM_QUANT_FN
    if _VLLM_QUANT_FN is not None:
        return _VLLM_QUANT_FN
    _ensure_vllm_baseline_shims()
    import triton
    import triton.language as tl

    @triton.jit
    def _per_token_quant_fp8(x_ptr, xq_ptr, scale_ptr, stride_x, stride_xq,
                             N, BLOCK: tl.constexpr, fp8_min, fp8_max):
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                    other=0.0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
        scale_x = absmax / fp8_max
        x_q = tl.clamp(x / scale_x, min=fp8_min, max=fp8_max).to(xq_ptr.dtype.element_ty)
        tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
        tl.store(scale_ptr + row_id, scale_x)

    def _quant(x, fp8type: int = 0):
        M = x.numel() // x.shape[-1]
        N = x.shape[-1]
        fp8_dt = torch.float8_e4m3fn if fp8type == 0 else torch.float8_e5m2
        fp8_max = 448.0 if fp8type == 0 else 57344.0
        x_q = torch.empty_like(x, device=x.device, dtype=fp8_dt)
        scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(N)
        num_warps = min(max(BLOCK // 256, 1), 8)
        finfo = torch.finfo(fp8_dt)
        _per_token_quant_fp8[(M,)](
            x, x_q, scales,
            stride_x=x.stride(-2), stride_xq=x_q.stride(-2),
            N=N, BLOCK=BLOCK, num_warps=num_warps, num_stages=1,
            fp8_min=finfo.min, fp8_max=fp8_max,
        )
        return x_q, scales

    _VLLM_QUANT_FN = _quant
    return _quant


def stage_benchmark(args: argparse.Namespace) -> None:
    global _VERBOSE_METRICS
    _VERBOSE_METRICS = True
    print("\n================ 3. BENCHMARK: Decode / Prefill / Skewed Gemma RMSNorm ================")
    from lightop import op as lightop_native
    from lightop.norm import (
        gemma_fused_add_rmsnorm,
        gemma_rms_norm_per_token_fp8_quant,
        gemma_rmsnorm,
    )

    baseline_kind = getattr(args, "baseline", "torch")
    use_vllm_baseline = False
    if baseline_kind == "vllm":
        try:
            _ensure_vllm_baseline_shims()
            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.model_executor.layers.layernorm import RMSNorm
            with set_current_vllm_config(VllmConfig()):
                _probe = RMSNorm(16, 1e-6).to(dtype=torch.float16, device="cuda")
            del _probe
            use_vllm_baseline = True
            print("[bench] vLLM/triton unfused baseline: ENABLED")
        except Exception as _e:
            use_vllm_baseline = False
            print(f"[bench] WARN: vLLM baseline unavailable "
                  f"({type(_e).__name__}: {_e}); using pure-PyTorch FP32 baseline")
    print(f"[bench] Primary baseline: {'vLLM/triton unfused' if use_vllm_baseline else 'pure-PyTorch FP32 eager'}; "
          f"repo-style allclose metrics: ON")

    eps = 1e-6
    copy_bw_gbs = measure_copy_bandwidth_gbs(
        warmup=min(args.warmup, 10), iters=min(args.iters, 50)
    )
    if copy_bw_gbs is not None:
        print(f"[bench] Calibrated D2D HBM Copy Bandwidth Ceiling: {copy_bw_gbs:.2f} GB/s")

    rows: list[dict[str, object]] = []
    hdr = (
        f"{'Path / Op':<26} | {'Regime':<8} | {'Shape (T,H)':<13} | {'Variant':<13} | "
        f"{'LightOp(us)':<11} | {'TorchRef(us)':<12} | {'Speedup':<8} | {'BW(GB/s)':<9} | {'%CopyBW':<8} | "
        f"{'VllmRef(us)':<11} | {'Spd/vllm':<9}"
    )

    def _record_and_print(
        op_name: str,
        regime: str,
        num_tokens: int,
        hidden: int,
        variant: str,
        lop_us: float,
        ref_us: float,
        logical_bytes: int,
        vllm_us: Optional[float] = None,
    ) -> None:
        bw_gbs = logical_bytes / (lop_us * 1e-6) / 1e9
        pct_bw = (100.0 * bw_gbs / copy_bw_gbs) if copy_bw_gbs else None
        pct_str = f"{pct_bw:6.1f}%" if pct_bw is not None else "   N/A"
        toks_per_ms = (num_tokens / (lop_us * 1e-3)) / 1000.0
        vllm_str = f"{vllm_us:.2f}" if vllm_us is not None else "N/A"
        spd_vllm = (vllm_us / lop_us) if vllm_us else None
        spd_vllm_str = f"{spd_vllm:.2f}x" if spd_vllm is not None else "N/A"
        print(
            f"{op_name:<26} | {regime:<8} | {f'{num_tokens}x{hidden}':<13} | {variant:<13} | "
            f"{lop_us:<11.2f} | {ref_us:<12.2f} | {f'{ref_us / lop_us:.2f}x':<8} | "
            f"{bw_gbs:<9.2f} | {pct_str:<8} | {vllm_str:<11} | {spd_vllm_str:<9}"
        )
        rows.append(
            {
                "op": op_name,
                "regime": regime,
                "tokens": num_tokens,
                "hidden": hidden,
                "variant": variant,
                "logical_bytes": logical_bytes,
                "lightop_us": round(lop_us, 3),
                "baseline_us": round(ref_us, 3),
                "speedup_vs_baseline": round(ref_us / lop_us, 3),
                "eff_bw_gbs": round(bw_gbs, 2),
                "pct_copy_bw": round(pct_bw, 2) if pct_bw is not None else "",
                "toks_per_ms": round(toks_per_ms, 2),
                "vllm_baseline_us": round(vllm_us, 3) if vllm_us is not None else "",
                "speedup_vs_vllm": round(spd_vllm, 3) if spd_vllm is not None else "",
            }
        )

    print(hdr)
    print("-" * len(hdr))

    # 3.1 Path 1: rms_norm_opt / gemma_rmsnorm (Decode / Prefill / Skewed)
    p1_bench = [
        # (regime, tokens, hidden, dtype, gemma, skewed)
        ("Decode",  1,     128,  torch.bfloat16, True,  False),
        ("Decode",  32,    128,  torch.bfloat16, True,  False),
        ("Decode",  1,     4096, torch.bfloat16, True,  False),
        ("Decode",  32,    3584, torch.bfloat16, True,  False),
        ("Decode",  288,   4096, torch.bfloat16, True,  False),
        ("Prefill", 4096,  128,  torch.bfloat16, True,  False),
        ("Prefill", 4096,  2560, torch.bfloat16, True,  False),
        ("Prefill", 4096,  3584, torch.bfloat16, True,  False),
        ("Prefill", 4096,  4096, torch.bfloat16, True,  False),
        ("Prefill", 4096,  4096, torch.float16,  True,  False),
        ("Prefill", 4096,  4096, torch.bfloat16, False, False),
        ("Prefill", 4096,  8192, torch.bfloat16, True,  False),
        ("Prefill", 16384, 4096, torch.bfloat16, True,  False),
        ("Skewed",  4096,  520,  torch.bfloat16, True,  True),
    ]
    for regime, nt, h, dt, gemma, skewed in p1_bench:
        x = torch.randn((nt, h), device="cuda", dtype=dt)
        if skewed:
            x[0::32].zero_()
            x[1::32] *= 100.0
        w = torch.randn((h,), device="cuda", dtype=dt)
        out = torch.empty_like(x)
        if gemma:
            gemma_rmsnorm(x, w, eps, out=out)
        else:
            lightop_native.rms_norm_opt(out, x, w, eps, -1.0)
        torch.cuda.synchronize()
        assert_path1_matches_ref(x, w, out, eps=eps, gemma=gemma, limit=-1.0, tag=f"b_p1_{nt}x{h}")

        iters = args.iters if nt <= 4096 else max(40, args.iters // 4)
        if gemma:
            lop_us = measure_call_us(lambda: lightop_native.gemma_rmsnorm(out, x, w, eps), args.warmup, iters)
        else:
            lop_us = measure_call_us(lambda: lightop_native.rms_norm_opt(out, x, w, eps, -1.0), args.warmup, iters)
        ref_us = measure_call_us(lambda: ref_rms_norm(x, w, eps, gemma=gemma), args.warmup, iters)
        vllm_us = None
        if use_vllm_baseline:
            vlayer = _get_vllm_layer(h, eps, dt, w)
            if gemma:
                vlayer.weight.data.add_(1.0)  # simulate Gemma (weight + 1)
            vllm_us = measure_call_us(lambda: vlayer.forward_cuda(x, None), args.warmup, iters)
            if gemma:
                vlayer.weight.data.sub_(1.0)
        logical_bytes = (2 * nt * h + h) * x.element_size()
        dt_tag = str(dt).removeprefix("torch.")[:4]
        op_lbl = "gemma_rmsnorm" if gemma else "rms_norm_opt"
        _record_and_print(op_lbl, regime, nt, h, f"{dt_tag}{'/skw' if skewed else ''}", lop_us, ref_us, logical_bytes, vllm_us=vllm_us)

    # 3.2 Path 2: gemma_fused_add_rmsnorm (Decode / Prefill / Skewed)
    p2_bench = [
        # (regime, tokens, hidden, dtype, skewed)
        ("Decode",  1,     128,  torch.bfloat16, False),
        ("Decode",  32,    128,  torch.bfloat16, False),
        ("Decode",  1,     4096, torch.bfloat16, False),
        ("Decode",  32,    4096, torch.bfloat16, False),
        ("Decode",  288,   4096, torch.bfloat16, False),
        ("Prefill", 4096,  128,  torch.bfloat16, False),
        ("Prefill", 4096,  2560, torch.bfloat16, False),
        ("Prefill", 4096,  3584, torch.bfloat16, False),
        ("Prefill", 4096,  4096, torch.bfloat16, False),
        ("Prefill", 4096,  4096, torch.float16,  False),
        ("Prefill", 4096,  8192, torch.bfloat16, False),
        ("Prefill", 16384, 4096, torch.bfloat16, False),
        ("Skewed",  4096,  4096, torch.bfloat16, True),
    ]
    for regime, nt, h, dt, skewed in p2_bench:
        x0 = torch.randn((nt, h), device="cuda", dtype=dt)
        r0 = torch.randn_like(x0)
        if skewed:
            r0[0::16].copy_((-x0[0::16].float() + 0.003).to(dt))
            x0[1::16].zero_()
        w = torch.randn((h,), device="cuda", dtype=dt)
        x_chk, r_chk = x0.clone(), r0.clone()
        gemma_fused_add_rmsnorm(x_chk, r_chk, w, eps)
        torch.cuda.synchronize()
        assert_path2_matches_ref(x0, r0, w, x_chk, r_chk, eps=eps, gemma=True, limit=-1.0, tag=f"b_p2_{nt}x{h}")

        iters = args.iters if nt <= 4096 else max(40, args.iters // 4)
        x_run, r_run = x0.clone(), r0.clone()
        # Keep magnitudes bounded across repeated in-place timing iterations
        x_run.mul_(0.01)
        r_run.mul_(0.01)
        lop_us = measure_call_us(
            lambda: lightop_native.gemma_fused_add_rmsnorm(x_run, r_run, w, eps),
            args.warmup, iters,
        )
        ref_us = measure_call_us(
            lambda: ref_fused_add_rms_norm(x0, r0, w, eps, gemma=True),
            args.warmup, iters,
        )
        vllm_us = None
        if use_vllm_baseline:
            vlayer = _get_vllm_layer(h, eps, dt, w)
            if gemma:
                vlayer.weight.data.add_(1.0)
            vx, vr = x0.clone(), r0.clone()
            vllm_us = measure_call_us(lambda: vlayer.forward_cuda(vx, vr)[0], args.warmup, iters)
            if gemma:
                vlayer.weight.data.sub_(1.0)
        logical_bytes = (4 * nt * h + h) * x0.element_size()
        dt_tag = str(dt).removeprefix("torch.")[:4]
        _record_and_print(
            "gemma_fused_add_rmsnorm", regime, nt, h,
            f"{dt_tag}{'/cancel' if skewed else ''}", lop_us, ref_us, logical_bytes,
            vllm_us=vllm_us,
        )

    # 3.3 Path 3: gemma_rms_norm_per_token_fp8_quant (Decode / Prefill / Skewed + Padded)
    p3_bench = [
        # (regime, tokens, hidden, dtype, fp8type, with_res, update_input, pad, skewed)
        ("Decode",  1,     4096, torch.bfloat16, 0, False, False, 0,  False),
        ("Decode",  32,    4096, torch.bfloat16, 0, False, False, 0,  False),
        ("Decode",  32,    4096, torch.bfloat16, 0, True,  False, 0,  False),
        ("Decode",  140,   4096, torch.bfloat16, 0, True,  False, 0,  False),
        ("Prefill", 4096,  1024, torch.bfloat16, 0, False, False, 0,  False),
        ("Prefill", 4096,  2560, torch.bfloat16, 0, False, False, 0,  False),
        ("Prefill", 4096,  3584, torch.bfloat16, 0, True,  False, 0,  False),
        ("Prefill", 4096,  4096, torch.bfloat16, 0, False, False, 0,  False),
        ("Prefill", 4096,  4096, torch.bfloat16, 0, True,  False, 0,  False),
        ("Prefill", 4096,  4096, torch.bfloat16, 0, True,  True,  0,  False),
        ("Prefill", 4096,  4096, torch.float16,  0, True,  False, 0,  False),
        ("Prefill", 4096,  4096, torch.bfloat16, 1, True,  False, 0,  False),
        ("Prefill", 4096,  8192, torch.bfloat16, 0, True,  False, 0,  False),
        ("Prefill", 16384, 4096, torch.bfloat16, 0, False, False, 0,  False),
        ("Skewed",  4096,  4096, torch.bfloat16, 0, True,  False, 0,  True),
        ("Skewed",  4096,  4096, torch.bfloat16, 0, False, False, 16, False),
    ]
    for regime, nt, h, dt, fp8type, with_res, upd, pad, skewed in p3_bench:
        if pad > 0:
            x0 = torch.randn((nt, h + pad), device="cuda", dtype=dt)[:, :h]
        else:
            x0 = torch.randn((nt, h), device="cuda", dtype=dt)
        r0 = torch.randn_like(x0) if with_res else None
        if skewed:
            x0[0::32].zero_()
            if r0 is not None:
                r0[0::32].zero_()
                r0[1::32].copy_((-x0[1::32].float() + 0.003).to(dt))
            x0[2::32] *= 100.0
        w = torch.randn((h,), device="cuda", dtype=dt)

        # Pre-check accuracy before timing
        x_chk = x0.clone()
        r_chk = r0.clone() if r0 is not None else None
        oq_chk, os_chk = gemma_rms_norm_per_token_fp8_quant(
            x_chk, w, epsilon=eps, fp8type=fp8type, residual=r_chk, update_input=upd
        )
        torch.cuda.synchronize()
        assert_path3_matches_ref(
            x0, r0, w, x_chk, r_chk, oq_chk, os_chk,
            eps=eps, fp8type=fp8type, gemma=True, update_input=upd,
            tag=f"b_p3_{nt}x{h}_res={with_res}_pad={pad}",
        )

        # Preallocate out_q & scales to time native C++ kernel dispatch cleanly (matches Group 1)
        fp8_dt = torch.float8_e4m3fn if fp8type == 0 else torch.float8_e5m2
        out_u8 = torch.empty((nt, h), device="cuda", dtype=fp8_dt).view(torch.uint8)
        scales = torch.empty((nt, 1), device="cuda", dtype=torch.float32)
        x_run = x0.clone() * 0.01 if upd else x0
        r_run = (r0.clone() * 0.01) if r0 is not None else None

        iters = args.iters if nt <= 4096 else max(40, args.iters // 4)
        lop_us = measure_call_us(
            lambda: lightop_native.gemma_rms_norm_fp8_quant(
                out_u8, x_run, w, scales, eps, r_run, upd, fp8type
            ),
            args.warmup, iters,
        )
        ref_us = measure_call_us(
            lambda: ref_rms_norm_per_token_fp8_quant(
                x0, w, eps, fp8type=fp8type, residual=r0, gemma=True
            ),
            args.warmup, iters,
        )

        vllm_us = None
        if use_vllm_baseline:
            vlayer = _get_vllm_layer(h, eps, dt, w)
            vlayer.weight.data.add_(1.0)  # Path 3 is always gemma=True
            vquant = _get_vllm_quant_fn()
            if r0 is None:
                vfwd = lambda: vquant(vlayer.forward_cuda(x0, None), fp8type)
            else:
                vfwd = lambda: vquant(vlayer.forward_cuda(x0, r0)[0], fp8type)
            vllm_us = measure_call_us(vfwd, args.warmup, iters)
            vlayer.weight.data.sub_(1.0)
        es = x0.element_size()
        logical_bytes = (
            nt * h * es  # read x
            + h * es  # read weight
            + nt * h  # write fp8 out
            + nt * 4  # write fp32 scales
            + (2 * nt * h * es if with_res else 0)  # read + write residual
            + (nt * h * es if upd else 0)  # optional write updated input
        )
        dt_tag = str(dt).removeprefix("torch.")[:4]
        fp8_tag = "e4m3" if fp8type == 0 else "e5m2"
        flags = ("R" if with_res else "N") + ("U" if upd else "") + (f"P{pad}" if pad else "") + ("S" if skewed else "")
        _record_and_print(
            "gemma_rmsnorm_fp8_quant", regime, nt, h,
            f"{dt_tag}/{fp8_tag}/{flags}", lop_us, ref_us, logical_bytes,
            vllm_us=vllm_us,
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
# Pytest Entrypoints (compatible with `pytest test/test_gemma_rmsnorm_gfx946.py`)
# =============================================================================

def test_gemma_rmsnorm_gfx946_isa_audit() -> None:
    import pytest

    if not _ensure_torch(optional=True) or not torch.cuda.is_available():
        pytest.skip("需要 PyTorch HIP 与 CUDA/ROCm 设备")
    maybe_add_local_build_to_syspath()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = props.gcnArchName.split(":", 1)[0]
    ns = argparse.Namespace(
        so_path=None,
        require_gfx946=False,
        strict_isa=(arch == "gfx946"),
        skip_isa=False,
        isa_only=False,
    )
    stage_env_and_isa_check(ns)


def test_gemma_rmsnorm_gfx946_correctness() -> None:
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
        description="gfx946 verification, ISA audit & benchmark for Gemma RMSNorm paths"
    )
    parser.add_argument("--so-path", type=str, default=None, help="Explicit path to compiled lightop op*.so")
    parser.add_argument("--require-gfx946", action="store_true", help="Fail if active GPU is not gfx946")
    parser.add_argument("--strict-isa", action="store_true", help="Fail if dccobjdump or expected gfx946 ISA hits are missing")
    parser.add_argument("--skip-isa", action="store_true", help="Skip dccobjdump / readelf ISA inspection")
    parser.add_argument("--isa-only", action="store_true", help="Only run Stage 1 environment and ISA inspection")
    parser.add_argument("--skip-bench", action="store_true", help="Skip performance benchmark stage")
    parser.add_argument("--perfmodel", action="store_true", help="Preset for slow PerfModel (--quick + --skip-bench)")
    parser.add_argument("--quick", action="store_true", help="Smaller accuracy matrix and skip large stress tests")
    parser.add_argument("--include-large-stress", action="store_true", help="Run int64 stress even with --quick")
    parser.add_argument("--stress-tokens", type=int, default=32768, help="Tokens for large-batch stress (default: 32768)")
    parser.add_argument("--xcd-loops", type=int, default=20, help="4-XCD consistency stress loop count (default: 20)")
    parser.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations (default: 20)")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark timed iterations (default: 200)")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path for benchmark results")
    parser.add_argument("--baseline", choices=["torch", "vllm"], default="torch",
                        help="Benchmark baseline: pure-PyTorch FP32 eager (default) or "
                             "vLLM RMSNorm (+triton FP8 for Path 3) unfused; lazy-loaded")
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
    if _ensure_torch(optional=bool(args.isa_only and args.so_path)):
        maybe_add_local_build_to_syspath()
    stage_env_and_isa_check(args)
    if args.isa_only:
        print("\nDONE (--isa-only).")
        return 0
    stage_correctness_and_edge_cases(args)
    if not args.skip_bench:
        stage_benchmark(args)
    print("\nALL GFX946 GEMMA RMSNORM VERIFICATION STAGES COMPLETED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
