#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""gfx946 verification, ISA audit & benchmark suite for ``per_token_group_quant_fp8``.

Covers:
1. Environment detection (rocminfo / hy-smi / torch / lightop.op).
2. Binary Code Object & ISA inspection via ``dccobjdump`` / ``readelf`` / ``hipprof`` on ``op*.so``
   (checks gfx946 target presence, SCRATCH == 0, ``v_cvt_scale_pk_{fp8,bf8}_*``,
   ``v_cvt_pkrp_e8m0_f32``, ``v_cvt_pk_f32_e8m0``, ``v_pk_max_{bf16,f16}``,
   ``v_permlane{16,32}_swap_b32`` distinct VGPRs, and zero SPF/mov_preserve regressions).
3. Comprehensive numerical accuracy & edge-case tests across BF16/FP16/FP32, E4M3/E5M2,
   group_size {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}, use_ue8m0 {False, True},
   all-zero groups with eps=0.0, exact 2^k * FP8_MAX boundaries, saturation, non-contiguous
   inputs, preallocated buffers, 3D shapes, and large int64 indexing stress.
4. Multi-shape latency & effective HBM bandwidth benchmark comparing:
   Triton (ue8m0=False/True) vs LightOp (use_ue8m0=False) vs LightOp (use_ue8m0=True).

Usage (from lightop repo root):
  python3 test/test_per_token_group_quant_fp8_gfx946.py
  python3 test/test_per_token_group_quant_fp8_gfx946.py --perfmodel
  python3 test/test_per_token_group_quant_fp8_gfx946.py --require-gfx946 --strict-isa --csv results_946.csv
  pytest -v test/test_per_token_group_quant_fp8_gfx946.py
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
import time
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


def fp8_ordinal(t: torch.Tensor) -> torch.Tensor:
    """Convert sign-magnitude FP8 bytes to signed ordinal integer for +0.0/-0.0-safe ULP diff."""
    u = t.view(torch.uint8).to(torch.int16)
    mag = u & 0x7F
    neg = (u & 0x80) != 0
    return torch.where(neg, -mag, mag)


def compute_exact_reference(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_dtype: torch.dtype,
    use_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
    x_f32 = x.float().reshape(-1, group_size)
    absmax = torch.clamp(x_f32.abs().amax(dim=-1), min=eps)
    scale_raw = absmax / fp8_max
    if use_ue8m0:
        ref_s = torch.zeros_like(scale_raw)
        pos = scale_raw > 0
        ref_s[pos] = torch.exp2(torch.ceil(torch.log2(scale_raw[pos])))
    else:
        ref_s = scale_raw

    s_exp = ref_s.unsqueeze(-1)
    scaled = torch.where(s_exp > 0, x_f32 / s_exp, torch.zeros_like(x_f32))
    ref_q = torch.clamp(scaled, -fp8_max, fp8_max).to(fp8_dtype).reshape(x.shape)
    expected_scale_shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    return ref_q, ref_s.reshape(expected_scale_shape)


# =============================================================================
# Stage 1: Environment & Binary ISA Self-Check (rocminfo / dccobjdump / readelf)
# =============================================================================

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

    # Linkage check (mirrors test_lmslim_native_linkage.py)
    if shutil.which("readelf"):
        rc_re, out_re = run_cmd(["readelf", "--dyn-syms", "--wide", so_path], timeout=30)
        if rc_re == 0:
            bad = [ln for ln in out_re.splitlines() if "__extendhfsf2" in ln and " UND " in ln]
            if bad:
                raise AssertionError("Unresolved __extendhfsf2 in .dynsym:\n" + "\n".join(bad))
            print("[readelf] OK: no unresolved __extendhfsf2 in .dynsym")

    # Resource metadata check via hipprof --codeobj-analyze if available
    if shutil.which("hipprof"):
        rc_ca, out_ca = run_cmd(["hipprof", "--codeobj-analyze", so_path], timeout=120)
        if rc_ca == 0:
            quant_lines = [
                ln.strip()
                for ln in out_ca.splitlines()
                if "per_token_group_quant" in ln or "gfx946" in ln
            ]
            if quant_lines:
                print("[hipprof --codeobj-analyze excerpt]\n  " + "\n  ".join(quant_lines[:20]))

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

    isa_report = audit_dccobjdump_quant_isa(out_d, strict=args.strict_isa)
    print(isa_report)
    return info


def audit_dccobjdump_quant_isa(dump_text: str, strict: bool = False) -> str:
    """Parse dccobjdump output for gfx946 per_token_group_quant_fp8 kernels & verify ISA/resources."""
    blocks = re.split(
        r"(?=^[^\n]*per_token_group_quant_fp8[^\n]*:|\t\.protected\t)",
        dump_text,
        flags=re.MULTILINE,
    )
    quant_blocks = [b for b in blocks if "per_token_group_quant_fp8" in b[:400]]
    if not quant_blocks:
        msg = "No per_token_group_quant_fp8 kernel blocks found in dccobjdump output"
        if strict:
            raise AssertionError(msg)
        return f"[isa][warn] {msg}"

    expected_insns = {
        "v_cvt_scale_pk_fp8_bf16": 0,
        "v_cvt_scale_pk_fp8_f16": 0,
        "v_cvt_scale_pk_fp8_f32": 0,
        "v_cvt_scale_pk_bf8_bf16": 0,
        "v_cvt_scale_pk_bf8_f16": 0,
        "v_cvt_scale_pk_bf8_f32": 0,
        "v_cvt_pkrp_e8m0_f32": 0,
        "v_cvt_pk_f32_e8m0": 0,
        "v_pk_max_bf16": 0,
        "v_pk_max_f16": 0,
        "v_permlane16_swap_b32": 0,
        "v_permlane32_swap_b32": 0,
        "v_pk_mul_f32": 0,
    }
    scratch_violations: list[str] = []
    same_reg_swaps: list[str] = []
    forbidden_insns: list[str] = []
    audited = 0

    for blk in quant_blocks:
        has_946_signal = any(
            kw in blk
            for kw in (
                "v_cvt_scale_pk_fp8_",
                "v_cvt_scale_pk_bf8_",
                "v_cvt_pkrp_e8m0_f32",
                "v_pk_max_bf16",
                "gfx946",
            )
        )
        if not has_946_signal:
            continue
        audited += 1
        header = blk.splitlines()[0].strip()[:110]
        priv = re.search(r"private_segment_fixed_size\s*[:= ]\s*(\d+)|SCRATCH\D+(\d+)", blk)
        if priv:
            val = int(priv.group(1) or priv.group(2))
            if val != 0:
                scratch_violations.append(f"{header} (scratch={val})")
        for kw in expected_insns:
            cnt = blk.count(kw)
            expected_insns[kw] += cnt
        for bad_kw in ("v_log_f32", "v_exp_f32", "mov_preserve"):
            if bad_kw in blk:
                forbidden_insns.append(f"{header} ({bad_kw} x{blk.count(bad_kw)})")
        for swap_kw in ("v_permlane16_swap_b32", "v_permlane32_swap_b32"):
            for ln in blk.splitlines():
                if swap_kw in ln:
                    regs = re.findall(r"\bv\d+\b", ln)
                    if len(regs) >= 2 and regs[0] == regs[1]:
                        same_reg_swaps.append(f"{header}: {ln.strip()}")

    if audited == 0:
        msg = "Found quant blocks in dccobjdump, but none emitted gfx946 instructions (was op*.so built with gfx946?)"
        if strict:
            raise AssertionError(msg)
        return f"[isa][warn] {msg}"

    if scratch_violations:
        raise AssertionError("Non-zero scratch spill in quant kernels: " + "; ".join(scratch_violations))
    if same_reg_swaps:
        raise AssertionError("Same-register operand in permlane swap: " + "; ".join(same_reg_swaps))
    if forbidden_insns:
        raise AssertionError("Unexpected SPF/preserve regression in gfx946 quant kernels: " + "; ".join(forbidden_insns))

    missing = [
        k for k in ("v_cvt_scale_pk_fp8_bf16", "v_cvt_scale_pk_fp8_f16", "v_cvt_pkrp_e8m0_f32", "v_permlane16_swap_b32")
        if expected_insns[k] == 0
    ]
    if missing and strict:
        raise AssertionError(f"Missing expected gfx946 ISA instructions in op*.so: {missing}")

    summary_lines = [f"[isa] Audited {audited} gfx946 per_token_group_quant_fp8 kernel section(s) (SCRATCH=0, distinct permlane regs):"]
    for kw, cnt in expected_insns.items():
        summary_lines.append(f"  {kw:<26}: {cnt} hit(s)")
    if missing:
        summary_lines.append(f"  [warn] Missing instructions (may require dccobjdump --extract-elf or per-arch dump): {missing}")
    return "\n".join(summary_lines)


# =============================================================================
# Stage 2: Correctness, API Alignment & Hardware Edge Cases
# =============================================================================

def assert_quant_matches_ref(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_dtype: torch.dtype,
    use_ue8m0: bool,
    tag: str,
) -> None:
    from lightop.quant import per_token_group_quant_fp8

    q, s = per_token_group_quant_fp8(
        x, group_size=group_size, eps=eps, dtype=fp8_dtype, use_ue8m0=use_ue8m0
    )
    torch.cuda.synchronize()

    ref_q, ref_s = compute_exact_reference(x, group_size, eps, fp8_dtype, use_ue8m0)
    assert not torch.isnan(s).any(), f"[{tag}] NaN detected in output scales"
    assert not torch.isnan(q.float()).any(), f"[{tag}] NaN detected in output FP8 values"

    torch.testing.assert_close(s, ref_s, rtol=1e-5, atol=1e-7)
    if use_ue8m0:
        bits = s.view(torch.int32)
        non_zero = s != 0
        if non_zero.any():
            assert torch.all((bits[non_zero] & 0x7FFFFF) == 0), (
                f"[{tag}] Non-zero UE8M0 scale must have 0 mantissa bits"
            )

    ulp_diff = (fp8_ordinal(q) - fp8_ordinal(ref_q)).abs().max().item()
    assert ulp_diff <= 1, f"[{tag}] FP8 ULP diff {ulp_diff} > 1"


def stage_correctness_and_edge_cases(args: argparse.Namespace) -> None:
    print("\n================ 2. CORRECTNESS & HARDWARE EDGE CASES ================")
    from lightop import op as lightop_native
    from lightop.quant import per_token_group_quant_fp8

    # 2.1 Signature, preallocated buffers & non-contiguous input
    expected_params = [
        "x", "group_size", "eps", "dtype",
        "column_major_scales", "out_q", "out_scale", "use_ue8m0",
    ]
    actual_params = list(inspect.signature(per_token_group_quant_fp8).parameters)
    assert actual_params == expected_params, f"Signature mismatch: {actual_params}"

    x_pre = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    out_q = torch.empty_like(x_pre, dtype=torch.float8_e4m3fn)
    out_s = torch.empty((4, 2), device="cuda", dtype=torch.float32)
    q_ret, s_ret = per_token_group_quant_fp8(
        x_pre, 128, eps=1e-5, dtype=torch.float8_e4m3fn,
        out_q=out_q, out_scale=out_s, use_ue8m0=True,
    )
    assert q_ret is out_q and s_ret is out_s
    q_nat = torch.empty_like(out_q)
    s_nat = torch.empty_like(out_s)
    lightop_native.per_token_group_quant_fp8(q_nat, x_pre, s_nat, 128, 1e-5, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_q.float(), q_nat.float(), rtol=0, atol=0)
    torch.testing.assert_close(out_s, s_nat, rtol=0, atol=0)

    # Non-contiguous slice
    x_nc = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert not x_nc.is_contiguous()
    assert_quant_matches_ref(x_nc, 128, 1e-5, torch.float8_e4m3fn, True, "non_contig_ue8m0")
    print("  [PASS] API signature, preallocated buffers & non-contiguous input")

    # 2.2 Full matrix across dtypes, fp8_dtype, group_size, ue8m0
    torch.manual_seed(946)
    group_sizes = [16, 128, 512, 1024] if args.quick else [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    tokens = 8 if args.quick else 32
    matrix_cases = 0
    for input_dtype in (torch.bfloat16, torch.float16, torch.float32):
        for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for gs in group_sizes:
                hidden = max(gs * 2, 256)
                for use_ue8m0 in (False, True):
                    x = torch.randn((tokens, hidden), dtype=input_dtype, device="cuda") * 3.0
                    x[0].zero_()
                    x[1, :gs] = 1e-7
                    assert_quant_matches_ref(
                        x, gs, 1e-10, fp8_dtype, use_ue8m0,
                        f"matrix_{input_dtype}_{fp8_dtype}_gs{gs}_ue8m0={use_ue8m0}",
                    )
                    matrix_cases += 1
    print(f"  [PASS] Dtype x FP8 x GroupSize x UE8M0 matrix ({matrix_cases} cases)")

    # 2.3 3D batched shape [batch, seq, hidden]
    x_3d = torch.randn((2, 16, 256), device="cuda", dtype=torch.bfloat16)
    assert_quant_matches_ref(x_3d, 128, 1e-10, torch.float8_e4m3fn, True, "3d_bf16_ue8m0")
    assert_quant_matches_ref(x_3d.half(), 128, 1e-10, torch.float8_e4m3fn, False, "3d_fp16_fp32s")
    print("  [PASS] 3D batched shapes")

    # 2.4 Hardware edge cases (eps=0 all-zero, exact 2^k * FP8_MAX, saturation, -0.0)
    edge_cases = 0
    for input_dtype in (torch.bfloat16, torch.float16, torch.float32):
        for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            fp8_max = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
            for gs in (128, 1024):
                # (a) All-zero with eps=0.0 -> scale MUST be exact +0.0f, NOT 2^-127 denorm
                z = torch.zeros((4, gs * 2), device="cuda", dtype=input_dtype)
                q_z, s_z = per_token_group_quant_fp8(
                    z, gs, eps=0.0, dtype=fp8_dtype, use_ue8m0=True
                )
                torch.cuda.synchronize()
                assert torch.all(s_z == 0.0), f"Expected exact 0.0f scale for all-zero eps=0, got {s_z}"
                assert torch.all(q_z.float() == 0.0)
                edge_cases += 1

                # (b) Exact 2^k * fp8_max -> absmax / FP8_MAX == 2^k, cvt_pkrp_e8m0 MUST NOT bump to 2^(k+1)
                exps = (-8, -4, -1, 0) if (input_dtype == torch.float16 and fp8_dtype == torch.float8_e5m2) else (-4, -1, 0, 2)
                p = torch.zeros((len(exps), gs), device="cuda", dtype=input_dtype)
                for row_idx, exp_k in enumerate(exps):
                    val = (2.0 ** exp_k) * fp8_max
                    p[row_idx, 0] = val
                    p[row_idx, 1] = -val
                assert_quant_matches_ref(
                    p, gs, 0.0, fp8_dtype, True,
                    f"exact_pow2_{input_dtype}_{fp8_dtype}_gs{gs}",
                )
                edge_cases += 1

                # (c) Saturation & -0.0
                sat = torch.randn((4, gs), device="cuda", dtype=input_dtype) * 100.0
                sat[0, 0] = -0.0
                sat[1, 0] = 60000.0 if input_dtype != torch.bfloat16 or fp8_dtype == torch.float8_e5m2 else 10000.0
                sat[1, 1] = -sat[1, 0]
                assert_quant_matches_ref(
                    sat, gs, 1e-10, fp8_dtype, True,
                    f"sat_negzero_{input_dtype}_{fp8_dtype}_gs{gs}",
                )
                edge_cases += 1
    print(f"  [PASS] Hardware edge cases: eps=0 zero, exact 2^k*FP8_MAX, saturation ({edge_cases} cases)")

    # 2.5 Large batch / int64 indexing stress (skip in --quick / --perfmodel unless --include-large-stress)
    if (not args.quick or args.include_large_stress) and args.stress_tokens > 0:
        st = args.stress_tokens
        sh = 4096
        x_big = torch.randn((st, sh), device="cuda", dtype=torch.bfloat16)
        q_b, s_b = per_token_group_quant_fp8(
            x_big, 128, eps=1e-10, dtype=torch.float8_e4m3fn, use_ue8m0=True
        )
        torch.cuda.synchronize()
        for sl in (slice(0, 16), slice(st - 16, st)):
            rq, rs = compute_exact_reference(x_big[sl], 128, 1e-10, torch.float8_e4m3fn, True)
            torch.testing.assert_close(s_b[sl], rs, rtol=1e-5, atol=1e-7)
            assert (fp8_ordinal(q_b[sl]) - fp8_ordinal(rq)).abs().max().item() <= 1
        print(f"  [PASS] Large-batch int64 stress ({st}x{sh}, {(st * sh) / 1e6:.1f}M elems)")


# =============================================================================
# Stage 3: Triton vs LightOp (use_ue8m0=False vs True) Benchmark
# =============================================================================

def _try_build_triton_quant():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def _triton_per_token_group_quant_fp8(
        y_ptr,
        y_q_ptr,
        y_s_ptr,
        group_size,
        y_num_columns,
        y_row_stride,
        eps,
        fp8_min,
        fp8_max,
        use_ue8m0: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        groups_per_row = y_num_columns // group_size
        g_id = tl.program_id(0)
        row = g_id // groups_per_row
        row_g_id = g_id % groups_per_row

        y_ptr += (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) * group_size)
        y_q_ptr += g_id.to(tl.int64) * group_size
        y_s_ptr += g_id

        cols = tl.arange(0, BLOCK)
        mask = cols < group_size
        y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        scale_raw = _absmax / fp8_max
        y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + cols, y_q, mask=mask)
        tl.store(y_s_ptr, y_s)

    def triton_quant(
        x: torch.Tensor,
        group_size: int,
        eps: float = 1e-10,
        dtype: torch.dtype = torch.float8_e4m3fn,
        use_ue8m0: bool = False,
        out_q: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
        fp8_min = -fp8_max
        x_q = out_q if out_q is not None else torch.empty_like(x, dtype=dtype)
        s_shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = out_scale if out_scale is not None else torch.empty(s_shape, device=x.device, dtype=torch.float32)
        num_groups = x.numel() // group_size
        block = triton.next_power_of_2(group_size)
        num_warps = min(max(block // 256, 1), 8)
        _triton_per_token_group_quant_fp8[(num_groups,)](
            x, x_q, x_s, group_size, x.shape[-1], x.stride(-2),
            eps, fp8_min, fp8_max, use_ue8m0,
            BLOCK=block, num_warps=num_warps, num_stages=1,
        )
        return x_q, x_s

    return triton_quant


def measure_call_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def stage_benchmark(args: argparse.Namespace) -> None:
    print("\n================ 3. BENCHMARK: Triton vs LightOp (!UE8M0 vs UE8M0) ================")
    from lightop.quant import per_token_group_quant_fp8

    triton_fn = _try_build_triton_quant()
    if triton_fn is None:
        print("[bench] Triton unavailable; benchmarking LightOp use_ue8m0=False vs use_ue8m0=True only.")

    # (num_tokens, hidden, group_size, input_dtype, fp8_dtype)
    bench_shapes = [
        (1,     4096,  128,  torch.bfloat16, torch.float8_e4m3fn),
        (8,     7168,  128,  torch.bfloat16, torch.float8_e4m3fn),
        (32,    4096,  128,  torch.bfloat16, torch.float8_e4m3fn),
        (128,   7168,  128,  torch.bfloat16, torch.float8_e4m3fn),
        (1024,  4096,  128,  torch.bfloat16, torch.float8_e4m3fn),
        (4096,  7168,  128,  torch.bfloat16, torch.float8_e4m3fn),
        (16384, 4096,  128,  torch.bfloat16, torch.float8_e4m3fn),
        (65536, 128,   128,  torch.bfloat16, torch.float8_e4m3fn),
        (4096,  4096,  128,  torch.float16,  torch.float8_e4m3fn),
        (4096,  4096,  512,  torch.bfloat16, torch.float8_e4m3fn),
        (4096,  4096,  1024, torch.bfloat16, torch.float8_e4m3fn),
    ]

    header = (
        f"{'Shape (T,H,GS)':<22} | {'Dtype':<11} | {'Bytes/g':<7} | "
        f"{'Tri_F32(us)':<11} | {'LOP_F32(us)':<11} | {'Tri_UE8(us)':<11} | "
        f"{'LOP_UE8(us)':<11} | {'BW_UE8(GB/s)':<12} | {'vsTriUE8':<8} | {'vsLOP_F32':<9}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict[str, object]] = []
    eps = 1e-10
    for num_tokens, hidden, gs, in_dt, fp8_dt in bench_shapes:
        x = torch.randn((num_tokens, hidden), device="cuda", dtype=in_dt)
        out_q = torch.empty_like(x, dtype=fp8_dt)
        out_s = torch.empty((num_tokens, hidden // gs), device="cuda", dtype=torch.float32)

        num_groups = (num_tokens * hidden) // gs
        logical_bytes = num_tokens * hidden * x.element_size() + num_tokens * hidden + num_groups * 4

        assert_quant_matches_ref(x, gs, eps, fp8_dt, True, f"bench_{num_tokens}x{hidden}_gs{gs}")

        tri_f32_us = tri_ue8_us = None
        if triton_fn is not None:
            try:
                tri_f32_us = measure_call_us(
                    lambda: triton_fn(x, gs, eps=eps, dtype=fp8_dt, use_ue8m0=False, out_q=out_q, out_scale=out_s),
                    args.warmup, args.iters,
                )
                tri_ue8_us = measure_call_us(
                    lambda: triton_fn(x, gs, eps=eps, dtype=fp8_dt, use_ue8m0=True, out_q=out_q, out_scale=out_s),
                    args.warmup, args.iters,
                )
            except Exception as exc:
                print(f"[warn] Triton failed on ({num_tokens},{hidden},{gs}): {exc}")
                triton_fn = None

        lop_f32_us = measure_call_us(
            lambda: per_token_group_quant_fp8(
                x, gs, eps=eps, dtype=fp8_dt, out_q=out_q, out_scale=out_s, use_ue8m0=False
            ),
            args.warmup, args.iters,
        )
        lop_ue8_us = measure_call_us(
            lambda: per_token_group_quant_fp8(
                x, gs, eps=eps, dtype=fp8_dt, out_q=out_q, out_scale=out_s, use_ue8m0=True
            ),
            args.warmup, args.iters,
        )

        bw_ue8_gbs = logical_bytes / (lop_ue8_us * 1e-6) / 1e9
        vs_tri = f"{tri_ue8_us / lop_ue8_us:.2f}x" if tri_ue8_us else "N/A"
        vs_lop = f"{lop_f32_us / lop_ue8_us:.2f}x"
        dt_str = f"{str(in_dt).removeprefix('torch.')[:4]}/{str(fp8_dt).removeprefix('torch.float8_')}"
        print(
            f"{f'{num_tokens}x{hidden}x{gs}':<22} | {dt_str:<11} | {logical_bytes / num_groups:<7.0f} | "
            f"{f'{tri_f32_us:.2f}' if tri_f32_us else 'N/A':<11} | {lop_f32_us:<11.2f} | "
            f"{f'{tri_ue8_us:.2f}' if tri_ue8_us else 'N/A':<11} | {lop_ue8_us:<11.2f} | "
            f"{bw_ue8_gbs:<12.2f} | {vs_tri:<8} | {vs_lop:<9}"
        )
        rows.append(
            {
                "num_tokens": num_tokens,
                "hidden": hidden,
                "group_size": gs,
                "input_dtype": str(in_dt).removeprefix("torch."),
                "fp8_dtype": str(fp8_dt).removeprefix("torch."),
                "logical_bytes": logical_bytes,
                "triton_fp32_us": round(tri_f32_us, 3) if tri_f32_us else "",
                "lightop_fp32_us": round(lop_f32_us, 3),
                "triton_ue8m0_us": round(tri_ue8_us, 3) if tri_ue8_us else "",
                "lightop_ue8m0_us": round(lop_ue8_us, 3),
                "lightop_ue8m0_gbps": round(bw_ue8_gbs, 2),
                "speedup_vs_triton_ue8m0": round(tri_ue8_us / lop_ue8_us, 3) if tri_ue8_us else "",
                "speedup_vs_lightop_fp32": round(lop_f32_us / lop_ue8_us, 3),
            }
        )

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[bench] CSV saved -> {csv_path}")


# =============================================================================
# Pytest Entrypoints (compatible with `pytest test/test_per_token_group_quant_fp8_gfx946.py`)
# =============================================================================

def test_per_token_group_quant_fp8_gfx946_isa_audit() -> None:
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


def test_per_token_group_quant_fp8_gfx946_correctness() -> None:
    import pytest

    if not _ensure_torch(optional=True) or not torch.cuda.is_available():
        pytest.skip("需要 PyTorch HIP 与 CUDA/ROCm 设备")
    maybe_add_local_build_to_syspath()
    ns = argparse.Namespace(
        quick=True,
        include_large_stress=False,
        stress_tokens=0,
    )
    stage_correctness_and_edge_cases(ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gfx946 verification, ISA audit & benchmark for per_token_group_quant_fp8"
    )
    parser.add_argument("--so-path", type=str, default=None, help="Explicit path to compiled lightop op*.so")
    parser.add_argument("--require-gfx946", action="store_true", help="Fail if active GPU is not gfx946")
    parser.add_argument("--strict-isa", action="store_true", help="Fail if dccobjdump or expected gfx946 ISA hits are missing")
    parser.add_argument("--skip-isa", action="store_true", help="Skip dccobjdump / readelf ISA inspection")
    parser.add_argument("--isa-only", action="store_true", help="Only run Stage 1 environment and ISA inspection")
    parser.add_argument("--skip-bench", action="store_true", help="Skip performance benchmark stage")
    parser.add_argument("--perfmodel", action="store_true", help="Preset for slow PerfModel (--quick + --skip-bench)")
    parser.add_argument("--quick", action="store_true", help="Smaller accuracy matrix and skip stress tests")
    parser.add_argument("--include-large-stress", action="store_true", help="Run int64 stress even with --quick")
    parser.add_argument("--stress-tokens", type=int, default=65536, help="Tokens for large-batch stress (default: 65536)")
    parser.add_argument("--warmup", type=int, default=15, help="Benchmark warmup iterations (default: 15)")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark timed iterations (default: 100)")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path for benchmark results")
    args = parser.parse_args()
    if args.perfmodel:
        args.quick = True
        args.skip_bench = True
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
    print("\nALL GFX946 PER_TOKEN_GROUP_QUANT_FP8 VERIFICATION STAGES COMPLETED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
