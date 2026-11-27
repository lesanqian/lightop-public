#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""gfx946 verification, ISA audit & benchmark suite for MoE routing & alignment.

Covers ``lightop.op.topk_softmax`` and ``lightop.moe.moe_align_block_size_out``:
1. Environment detection (hy-smi / rocm-smi / rocminfo / torch / lightop.op).
2. Binary Code Object & ISA inspection via ``readelf`` / ``hipprof`` / ``dccobjdump``
   on ``op*.so``:
   - Verifies ``SCRATCH == 0`` on gfx946 ``topk*GatingSoftmax*`` and ``moe_align*`` kernels.
   - Verifies ``v_permlane{16,32}_swap_b32`` uses distinct VGPR operands (no same-reg hazard).
   - Verifies zero ``ds_bpermute_b32`` inside gfx946 ``topk8GatingSoftmax_packed16`` sections.
   - Tallies gfx946 instructions (``v_permlane{16,32}_swap_b32``, ``v_pk_max_{bf16,f16}``,
     ``v_pk_{mul,fma}_f32``, ``v_mov_b32_dpp``) and device-scope fence/atomic coherence.
3. Comprehensive numerical accuracy, edge cases & 4-XCD consistency stress:
   - API / runner contract parity (``lightop.op.topk_softmax`` vs ``lightop.moe.topk_softmax``,
     ``moe_align_block_size_out`` vs ``op.moe_align_block_size`` / ``moe_align_block_size``).
   - ``topk_softmax`` across BF16/FP16/FP32, E in {1..512, 160, 300}, K in {1,2,4,6,8},
     renormalize True/False, int32/int64, +/-0.0 tie-breaking, and T across 512/1024 cutoffs.
   - ``moe_align_block_size_out`` across E<=64, medium CUB matching, multi-block non-identity
     Is_EP, E>512, skewed hot-expert contention, non-pow2 block_size, num_local_tokens, and
     repeated 4-XCD multi-block consistency stress.
4. Focused Decode / Prefill / Skewed-routing benchmarks + optional CSV export.

Usage (from lightop repo root):
  python3 test/test_moe_routing_gfx946.py
  python3 test/test_moe_routing_gfx946.py --isa-only --strict-isa --so-path path/to/op.so
  python3 test/test_moe_routing_gfx946.py --perfmodel --require-gfx946 --strict-isa
  python3 test/test_moe_routing_gfx946.py --require-gfx946 --strict-isa --csv moe_routing_946.csv
  pytest -v test/test_moe_routing_gfx946.py
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
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

_TARGET_LINE_RE = re.compile(r"(?:amdgcn-amd-amdhsa--|mcpu=|arch=)(gfx\w+)|^\s*\.amdgcn_target\s+\"([^\"]+)\"")


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


def _check_block_scratch(blk: str, header: str, violations: list[str]) -> None:
    for m in re.finditer(
        r"(?:\.amdhsa_)?private_segment_fixed_size\s*(?:[:= ]\s*)(\d+)|SCRATCH\D+(\d+)",
        blk,
    ):
        val = int(m.group(1) or m.group(2))
        if val != 0:
            violations.append(f"{header} (scratch={val})")
            return


def audit_dccobjdump_moe_routing_isa(dump_text: str, strict: bool = False) -> str:
    """Parse dccobjdump output for gfx946 topk_softmax & moe_align_block_size kernels."""
    segments = _extract_arch_segments(dump_text)
    explicit_946_present = any(arch == "gfx946" for arch, _ in segments)

    topk_blocks: list[tuple[str, str]] = []
    align_blocks: list[tuple[str, str]] = []
    for seg_arch, seg_text in segments:
        if explicit_946_present and seg_arch != "gfx946":
            continue
        blocks = re.split(
            r"(?=^[^\n]*(?:topk\d*GatingSoftmax|moeSoftmax|moeTopK|moe_align_block_size|count_and_sort_expert_tokens)[^\n]*:|\t\.protected\t)",
            seg_text,
            flags=re.MULTILINE,
        )
        for b in blocks:
            if len(b.splitlines()) <= 1:
                continue
            head = b[:400]
            if any(k in head for k in ("topk2GatingSoftmax", "topk4GatingSoftmax", "topk8GatingSoftmax", "topkGatingSoftmax")):
                topk_blocks.append((seg_arch, b))
            elif any(k in head for k in ("moe_align_block_size_kernel", "count_and_sort_expert_tokens_kernel")):
                align_blocks.append((seg_arch, b))

    if not topk_blocks and not align_blocks:
        msg = "No topk_softmax or moe_align_block_size kernel blocks found in dccobjdump output"
        if strict:
            raise AssertionError(msg)
        return f"[isa][warn] {msg}"

    expected_topk_insns = {
        "v_permlane16_swap_b32": 0,
        "v_permlane32_swap_b32": 0,
        "v_pk_max_bf16": 0,
        "v_pk_max_f16": 0,
        "v_pk_mul_f32": 0,
        "v_pk_fma_f32": 0,
        "v_mov_b32_dpp": 0,
        "ds_bpermute_b32": 0,
    }
    scratch_violations: list[str] = []
    same_reg_swaps: list[str] = []
    packed_bpermute_regs: list[str] = []
    audited_topk = 0
    audited_align = 0
    align_fence_or_atomic_hits = 0

    for seg_arch, blk in topk_blocks:
        has_946_signal = (seg_arch == "gfx946") or any(
            kw in blk
            for kw in (
                "v_permlane16_swap_b32",
                "v_permlane32_swap_b32",
                "v_pk_max_bf16",
                "v_pk_max_f16",
                "gfx946",
            )
        )
        if not has_946_signal:
            continue
        audited_topk += 1
        header = blk.splitlines()[0].strip()[:120]
        _check_block_scratch(blk, header, scratch_violations)

        for kw in expected_topk_insns:
            expected_topk_insns[kw] += blk.count(kw)

        if "packed16" in header and "ds_bpermute_b32" in blk:
            packed_bpermute_regs.append(f"{header} (ds_bpermute x{blk.count('ds_bpermute_b32')})")

        for swap_kw in ("v_permlane16_swap_b32", "v_permlane32_swap_b32"):
            for ln in blk.splitlines():
                if swap_kw in ln:
                    regs = re.findall(r"\bv\d+\b", ln)
                    if len(regs) >= 2 and regs[0] == regs[1]:
                        same_reg_swaps.append(f"{header}: {ln.strip()}")

    for seg_arch, blk in align_blocks:
        if explicit_946_present and seg_arch != "gfx946":
            continue
        audited_align += 1
        header = blk.splitlines()[0].strip()[:120]
        _check_block_scratch(blk, header, scratch_violations)
        if "opt_cub" in header:
            for fence_kw in ("buffer_wbl2", "global_atomic", "buffer_atomic", "s_waitcnt"):
                align_fence_or_atomic_hits += blk.count(fence_kw)

    if audited_topk == 0:
        msg = (
            "Found MoE routing kernel blocks in dccobjdump, but none matched gfx946 "
            "signals (was op*.so built with gfx946?)"
        )
        if strict:
            raise AssertionError(msg)
        return f"[isa][warn] {msg}"

    if scratch_violations:
        raise AssertionError("Non-zero scratch spill in MoE routing kernels: " + "; ".join(scratch_violations))
    if same_reg_swaps:
        raise AssertionError("Same-register operand in permlane swap: " + "; ".join(same_reg_swaps))
    if packed_bpermute_regs:
        raise AssertionError(
            "Unexpected ds_bpermute_b32 in gfx946 packed16 kernels: "
            + "; ".join(packed_bpermute_regs)
        )

    missing = [
        k for k in ("v_permlane16_swap_b32", "v_permlane32_swap_b32", "v_pk_max_bf16", "v_pk_max_f16")
        if expected_topk_insns[k] == 0
    ]
    if missing and strict:
        raise AssertionError(f"Missing expected gfx946 ISA instructions in op*.so: {missing}")
    if audited_align == 0 and strict:
        raise AssertionError("No moe_align_block_size kernel sections audited in op*.so")

    lines = [
        f"[isa] Audited {audited_topk} gfx946 topk kernel(s) & {audited_align} moe_align kernel(s) "
        f"(SCRATCH=0, distinct permlane regs, packed16 ds_bpermute=0):"
    ]
    for kw, cnt in expected_topk_insns.items():
        lines.append(f"  {kw:<26}: {cnt} hit(s)")
    lines.append(f"  {'align_fence/atomic hits':<26}: {align_fence_or_atomic_hits} hit(s)")
    if missing:
        lines.append(
            f"  [warn] Missing expected instructions (may need per-arch extraction): {missing}"
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
            moe_lines = [
                ln.strip()
                for ln in out_ca.splitlines()
                if any(k in ln for k in ("topk", "moe_align", "count_and_sort", "gfx946"))
            ]
            if moe_lines:
                print("[hipprof --codeobj-analyze excerpt]\n  " + "\n  ".join(moe_lines[:24]))

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

    print(audit_dccobjdump_moe_routing_isa(out_d, strict=args.strict_isa))
    return info


# =============================================================================
# Reference & Validation Helpers
# =============================================================================

def assert_topk_matches_ref(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    source_rows: Optional[torch.Tensor],
    renormalize: bool,
    tag: str,
) -> float:
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    if num_tokens == 0:
        return 0.0

    assert not torch.isnan(topk_weights).any(), f"[{tag}] NaN in topk_weights"
    assert torch.all((topk_indices >= 0) & (topk_indices < num_experts)), f"[{tag}] OOB expert index"
    if topk > 1:
        sorted_idx = torch.sort(topk_indices.to(torch.int64), dim=-1).values
        assert torch.all(sorted_idx[:, 1:] != sorted_idx[:, :-1]), f"[{tag}] Duplicate expert IDs"

    logits_f32 = gating_output.float()
    selected_logits = torch.gather(logits_f32, 1, topk_indices.to(torch.int64))
    selected_sorted = torch.sort(selected_logits, dim=-1, descending=True).values
    ref_logits = torch.topk(logits_f32, topk, dim=-1, sorted=True).values
    assert torch.equal(selected_sorted, ref_logits), f"[{tag}] Selected top-k logits mismatch"

    if renormalize:
        ref_weights = torch.softmax(selected_logits, dim=-1)
    else:
        ref_weights = torch.gather(torch.softmax(logits_f32, dim=-1), 1, topk_indices.to(torch.int64))

    max_err = (topk_weights - ref_weights).abs().max().item()
    torch.testing.assert_close(
        topk_weights,
        ref_weights,
        rtol=2e-4,
        atol=2e-5,
        msg=f"[{tag}] weight error={max_err:.6e}",
    )

    if source_rows is not None:
        rows = torch.arange(num_tokens, device=gating_output.device)[:, None]
        ranks = torch.arange(topk, device=gating_output.device)[None, :]
        expected_src = (ranks * num_tokens + rows).to(source_rows.dtype)
        assert torch.equal(source_rows, expected_src), f"[{tag}] source_rows mismatch"

    return max_err


def reference_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    expert_map: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[int] = None,
    is_ep: bool = False,
) -> tuple[list[list[int]], list[int], int, list[bool]]:
    flat = topk_ids.reshape(-1).cpu().tolist()
    topk = topk_ids.size(-1) if topk_ids.dim() >= 2 else 8
    eff_numel = len(flat) if num_local_tokens is None else num_local_tokens * topk
    emap = expert_map.cpu().tolist() if expert_map is not None else None

    if num_local_tokens == 0:
        return [], [], 0, []

    buckets: list[list[int]] = [[] for _ in range(num_experts)]
    mask: list[bool] = []
    for idx in range(eff_numel):
        eid = flat[idx]
        valid = (emap is None) or (emap[eid] != -1)
        mask.append(valid)
        if not is_ep or valid:
            buckets[eid].append(idx)

    expected_buckets: list[list[int]] = []
    expected_expert_ids: list[int] = []
    total_post_pad = 0
    for eid in range(num_experts):
        cnt = len(buckets[eid])
        padded = math.ceil(cnt / block_size) * block_size
        nblocks = padded // block_size
        mapped = emap[eid] if emap is not None else eid
        for _ in range(nblocks):
            expected_expert_ids.append(mapped)
        if padded > 0:
            expected_buckets.append(sorted(buckets[eid]))
        total_post_pad += padded
    return expected_buckets, expected_expert_ids, total_post_pad, mask


def assert_align_matches_ref(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    expert_mask: Optional[torch.Tensor],
    num_local_tokens: Optional[int],
    is_ep: bool,
    is_fuse_fill: bool,
    tag: str,
) -> None:
    ref_buckets, ref_eids, ref_total, ref_mask = reference_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        expert_map=expert_map,
        num_local_tokens=num_local_tokens,
        is_ep=is_ep,
    )
    got_total = int(num_tokens_post_pad.item())
    assert got_total == ref_total, f"[{tag}] total_post_pad={got_total} != ref={ref_total}"

    if got_total == 0:
        return

    got_eids = expert_ids[: got_total // block_size].cpu().tolist()
    assert got_eids == ref_eids, f"[{tag}] expert_ids mismatch"

    if expert_mask is not None and num_local_tokens is None:
        assert expert_mask.cpu().tolist() == ref_mask, f"[{tag}] expert_mask mismatch"

    numel = topk_ids.numel()
    sorted_cpu = sorted_ids.cpu().tolist()
    if is_fuse_fill:
        assert all(v == numel for v in sorted_cpu[got_total:]), f"[{tag}] tail padding sentinel corrupted"

    offset = 0
    for expected_tokens in ref_buckets:
        padded = math.ceil(len(expected_tokens) / block_size) * block_size
        segment = sorted_cpu[offset : offset + padded]
        actual_tokens = sorted(v for v in segment if v != numel)
        pad_count = sum(1 for v in segment if v == numel)
        assert actual_tokens == expected_tokens, f"[{tag}] bucket tokens mismatch at offset {offset}"
        assert pad_count == padded - len(expected_tokens), f"[{tag}] pad count mismatch"
        offset += padded
    assert offset == got_total, f"[{tag}] consumed={offset} != got_total={got_total}"


# =============================================================================
# Stage 2: Correctness, API/Runner Contract & 4-XCD Consistency Stress
# =============================================================================

def stage_correctness_and_edge_cases(args: argparse.Namespace) -> None:
    print("\n================ 2. CORRECTNESS, EDGE CASES & 4-XCD STRESS ================")
    import lightop
    from lightop import op as lightop_native
    from lightop.moe import routing

    # 2.1 Runner & categorized API alignment
    assert routing.topk_softmax is lightop_native.topk_softmax
    assert lightop.moe.topk_softmax is lightop_native.topk_softmax
    align_params = list(inspect.signature(routing.moe_align_block_size_out).parameters)
    assert align_params == [
        "topk_ids", "num_experts", "block_size", "sorted_ids", "expert_ids",
        "num_tokens_post_pad", "expert_map", "expert_mask", "num_local_tokens",
        "is_ep", "is_fuse_fill",
    ], f"Unexpected moe_align_block_size_out params: {align_params}"
    print("  [PASS] Runner & categorized routing API contract alignment")

    # 2.2 topk_softmax full matrix (BF16/FP16/FP32, wide E, K, renorm, Optional source_rows)
    torch.manual_seed(946)
    if args.quick:
        topk_configs = [
            (0, 256, 8, torch.bfloat16, torch.int32, True, True),
            (1, 128, 8, torch.bfloat16, torch.int32, True, True),
            (64, 128, 8, torch.bfloat16, torch.int64, True, False),
            (1025, 128, 8, torch.float16, torch.int32, True, True),
            (512, 256, 8, torch.bfloat16, torch.int32, True, True),
            (1025, 256, 8, torch.bfloat16, torch.int64, True, False),
            (64, 256, 8, torch.float32, torch.int32, True, True),
            (64, 256, 8, torch.bfloat16, torch.int32, False, True),
            (16, 512, 8, torch.bfloat16, torch.int32, True, True),
            (1025, 512, 8, torch.bfloat16, torch.int64, True, False),
            (32, 512, 8, torch.float32, torch.int32, False, True),
            (32, 160, 8, torch.bfloat16, torch.int32, True, True),
            (32, 300, 8, torch.bfloat16, torch.int32, False, True),
            (32, 64, 4, torch.bfloat16, torch.int32, True, True),
            (32, 256, 4, torch.float32, torch.int32, False, True),
            (32, 64, 2, torch.bfloat16, torch.int32, True, True),
            (32, 256, 2, torch.float16, torch.int64, False, True),
            (32, 256, 6, torch.bfloat16, torch.int32, True, True),
            (16, 4, 2, torch.float16, torch.int32, True, True),
        ]
    else:
        topk_configs = [
            (t, e, k, dt, idt, ren, with_src)
            for (e, k) in (
                (1, 1), (2, 2), (4, 2), (8, 4), (32, 8), (64, 2), (64, 4), (64, 8),
                (128, 6), (128, 8), (160, 8), (256, 2), (256, 4), (256, 6), (256, 8),
                (512, 2), (512, 4), (512, 8),
            )
            for dt in (torch.bfloat16, torch.float16, torch.float32)
            for ren in (True, False)
            for (t, idt, with_src) in (
                (7, torch.int32, True),
                (513, torch.int64, False),
                (1025, torch.int32, True),
            )
        ] + [
            (32, 300, 8, dt, torch.int32, False, True)
            for dt in (torch.bfloat16, torch.float16, torch.float32)
        ] + [(0, 256, 8, torch.bfloat16, torch.int32, True, True)]

    for t, e, k, dt, idt, ren, with_src in topk_configs:
        logits = torch.randn((t, e), device="cuda", dtype=dt) * 2.5
        weights = torch.empty((t, k), device="cuda", dtype=torch.float32)
        indices = torch.empty((t, k), device="cuda", dtype=idt)
        src_rows = torch.empty((t, k), device="cuda", dtype=torch.int32) if with_src else None
        lightop_native.topk_softmax(weights, indices, src_rows, logits, ren)
        torch.cuda.synchronize()
        assert_topk_matches_ref(
            logits, weights, indices, src_rows, ren,
            f"topk_T{t}_E{e}_K{k}_{dt}_{idt}_ren{ren}",
        )

    # Signed-zero & equal-value tie-breaking across packed16 & opt paths
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        for e in (128, 256, 512):
            gating = torch.full((4, e), -10.0, device="cuda", dtype=dt)
            gating[:, 15] = 0.0
            gating[:, 9] = -0.0
            gating[:, 4] = 0.0
            gating[:, 1] = -0.0
            gating[:, 0] = 0.0
            w = torch.empty((4, 8), device="cuda", dtype=torch.float32)
            idx = torch.empty((4, 8), device="cuda", dtype=torch.int32)
            lightop_native.topk_softmax(w, idx, None, gating, True)
            torch.cuda.synchronize()
            exp_top5 = torch.tensor([0, 1, 4, 9, 15], device="cuda", dtype=torch.int32)
            for r in range(4):
                assert torch.equal(idx[r, :5], exp_top5), f"tie-break failed {dt} E={e}: {idx[r, :5]}"
    print(f"  [PASS] topk_softmax matrix ({len(topk_configs)} configs + tie-break checks)")

    # 2.3 moe_align_block_size_out matrix (bug fixes, 128 KB LDS multi-copy, EP, skew)
    align_cases = [
        # (num_tokens, topk, num_experts, block_size, dtype, is_ep, is_fuse_fill, skewed, ep_all_masked)
        (0,    8, 256, 16, torch.int32, False, True,  False, False),
        (8,    2, 32,  16, torch.int32, False, True,  False, False),
        (16,   4, 64,  16, torch.int64, True,  True,  False, False),
        (256,  8, 128, 16, torch.int32, False, True,  False, False),
        (256,  8, 256, 32, torch.int32, True,  True,  True,  False),
        (400,  8, 256, 16, torch.int64, True,  False, True,  False),
        (600,  8, 256, 16, torch.int32, True,  True,  True,  False),
        (512,  8, 256, 64, torch.int32, True,  True,  False, False),
        (1024, 8, 256, 16, torch.int32, True,  True,  False, True),
        (2560, 8, 256, 16, torch.int32, False, True,  True,  False),
        (2560, 8, 256, 16, torch.int64, True,  True,  False, False),
        (2560, 8, 256, 24, torch.int32, False, True,  False, False),
        (2560, 8, 600, 16, torch.int32, True,  True,  True,  False),
        (2560, 8, 1024, 16, torch.int32, False, True, False, False),
    ]
    for nt, k, ne, bs, dt, is_ep, fuse_fill, skewed, all_masked in align_cases:
        if nt == 0:
            topk_ids = torch.empty((0, k), device="cuda", dtype=dt)
        elif skewed:
            hot = torch.tensor([0, ne - 1], device="cuda", dtype=dt)
            topk_ids = hot[torch.randint(0, 2, (nt, k), device="cuda")]
        else:
            topk_ids = torch.randint(0, ne, (nt, k), device="cuda", dtype=dt)

        expert_map = None
        if is_ep or all_masked:
            expert_map = torch.full((ne,), -1, device="cuda", dtype=torch.int32)
            if not all_masked:
                # Non-identity EP mapping: interleave valid local shards to stress global->local map.
                local_ids = torch.arange(ne // 4, device="cuda", dtype=torch.int32)
                expert_map[ne - (ne // 4) :] = local_ids

        numel = topk_ids.numel()
        max_padded = numel + ne * (bs - 1) + 3
        max_blocks = max(math.ceil(max_padded / bs), 1)
        sorted_ids = (
            torch.empty((max_padded,), dtype=torch.int32, device="cuda")
            if fuse_fill
            else torch.full((max_padded,), numel, dtype=torch.int32, device="cuda")
        )
        expert_ids = torch.full((max_blocks,), -99, dtype=torch.int32, device="cuda")
        num_tokens_post_pad = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        expert_mask = torch.empty((numel,), dtype=torch.bool, device="cuda") if numel > 0 else None

        routing.moe_align_block_size_out(
            topk_ids, ne, bs, sorted_ids, expert_ids, num_tokens_post_pad,
            expert_map=expert_map, expert_mask=expert_mask,
            is_ep=is_ep, is_fuse_fill=fuse_fill,
        )
        torch.cuda.synchronize()
        assert_align_matches_ref(
            topk_ids, ne, bs, sorted_ids, expert_ids, num_tokens_post_pad,
            expert_map, expert_mask, None, is_ep, fuse_fill,
            f"align_T{nt}_K{k}_E{ne}_BS{bs}_{dt}_ep{is_ep}_skew{skewed}",
        )

    # num_local_tokens with non-8 topk & local_tok=0 across single-block & multi-block buffers
    for cap_tokens in (100, 3000):
        ne, k, bs = 256, 4, 16
        topk_ids = torch.randint(0, ne, (cap_tokens, k), device="cuda", dtype=torch.int32)
        max_padded = topk_ids.numel() + ne * (bs - 1)
        max_blocks = math.ceil(max_padded / bs)
        for local_tok in (0, 37):
            sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device="cuda")
            expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device="cuda")
            num_tokens_post_pad = torch.full((1,), -1, dtype=torch.int32, device="cuda")
            nlt = torch.tensor([local_tok], dtype=torch.int32, device="cuda")
            routing.moe_align_block_size_out(
                topk_ids, ne, bs, sorted_ids, expert_ids, num_tokens_post_pad,
                num_local_tokens=nlt, is_ep=False, is_fuse_fill=True,
            )
            torch.cuda.synchronize()
            assert_align_matches_ref(
                topk_ids, ne, bs, sorted_ids, expert_ids, num_tokens_post_pad,
                None, None, local_tok, False, True,
                f"nlt_cap{cap_tokens}_loc{local_tok}",
            )
    print(f"  [PASS] moe_align_block_size_out matrix ({len(align_cases)} cases + num_local_tokens)")

    # 2.4 End-to-end topk_softmax -> moe_align_block_size_out pipeline & 4-XCD consistency stress
    if (not args.quick or args.include_large_stress) and args.stress_tokens > 0:
        st = args.stress_tokens
        for ne in (256, 512):
            x_big = torch.randn((st, ne), device="cuda", dtype=torch.bfloat16)
            w_big = torch.empty((st, 8), device="cuda", dtype=torch.float32)
            idx_big = torch.empty((st, 8), device="cuda", dtype=torch.int32)
            lightop_native.topk_softmax(w_big, idx_big, None, x_big, True)
            torch.cuda.synchronize()
            for sl in (slice(0, 16), slice(st - 16, st)):
                assert_topk_matches_ref(x_big[sl], w_big[sl], idx_big[sl], None, True, f"stress_E{ne}")
        print(f"  [PASS] Large-batch topk_softmax int64 stress (T={st}, E=256/512)")

    xcd_tokens = max(2560, args.stress_tokens // 4) if not args.quick else 2560
    xcd_loops = args.xcd_loops
    ne, bs, k = 256, 16, 8
    emap_xcd = torch.full((ne,), -1, device="cuda", dtype=torch.int32)
    emap_xcd[ne - 64 :] = torch.arange(64, device="cuda", dtype=torch.int32)
    gating_xcd = torch.randn((xcd_tokens, ne), device="cuda", dtype=torch.bfloat16)
    gating_xcd[:, -8:] += 3.0  # Skew routing toward local hot experts
    w_xcd = torch.empty((xcd_tokens, k), device="cuda", dtype=torch.float32)
    idx_xcd = torch.empty((xcd_tokens, k), device="cuda", dtype=torch.int32)
    lightop_native.topk_softmax(w_xcd, idx_xcd, None, gating_xcd, True)
    torch.cuda.synchronize()

    max_padded = idx_xcd.numel() + ne * (bs - 1)
    max_blocks = math.ceil(max_padded / bs)
    ref_buckets, ref_eids, ref_total, _ = reference_moe_align_block_size(
        idx_xcd, ne, bs, expert_map=emap_xcd, is_ep=True
    )
    sorted_buf = torch.empty((max_padded,), dtype=torch.int32, device="cuda")
    eids_buf = torch.empty((max_blocks,), dtype=torch.int32, device="cuda")
    tot_buf = torch.empty((1,), dtype=torch.int32, device="cuda")
    for it in range(xcd_loops):
        eids_buf.fill_(-77)
        tot_buf.fill_(-1)
        routing.moe_align_block_size_out(
            idx_xcd, ne, bs, sorted_buf, eids_buf, tot_buf,
            expert_map=emap_xcd, is_ep=True, is_fuse_fill=True,
        )
        torch.cuda.synchronize()
        got_tot = int(tot_buf.item())
        assert got_tot == ref_total, f"[4xcd loop {it}] total={got_tot} != {ref_total}"
        assert eids_buf[: got_tot // bs].cpu().tolist() == ref_eids, f"[4xcd loop {it}] expert_ids mismatch"
    assert_align_matches_ref(
        idx_xcd, ne, bs, sorted_buf, eids_buf, tot_buf,
        emap_xcd, None, None, True, True, "4xcd_e2e_final",
    )
    print(f"  [PASS] 4-XCD multi-block consistency stress ({xcd_loops}x iters, T={xcd_tokens}, K={k}, E={ne})")


# =============================================================================
# Stage 3: Focused Benchmark + CSV Export
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


def _torch_topk_softmax_ref(
    gating_output: torch.Tensor, topk: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if renormalize:
        vals, idx = torch.topk(gating_output.float(), topk, dim=-1)
        return torch.softmax(vals, dim=-1), idx
    probs = torch.softmax(gating_output.float(), dim=-1)
    return torch.topk(probs, topk, dim=-1)


def measure_copy_bandwidth_gbs(warmup: int = 10, iters: int = 50) -> Optional[float]:
    """Calibrate practical D2D HBM copy bandwidth ceiling (Read + Write = 2x buffer)."""
    try:
        elems = 32 * 1024 * 1024  # 128 MiB per tensor (total 256 MiB traffic)
        src = torch.empty((elems,), device="cuda", dtype=torch.float32)
        dst = torch.empty((elems,), device="cuda", dtype=torch.float32)
        src.fill_(1.0)
        us = measure_call_us(lambda: dst.copy_(src), warmup, iters)
        nbytes = 2 * elems * 4
        return nbytes / (us * 1e-6) / 1e9
    except RuntimeError:
        return None


def stage_benchmark(args: argparse.Namespace) -> None:
    print("\n================ 3. BENCHMARK: Decode / Prefill / Skewed MoE Routing ================")
    from lightop import op as lightop_native
    from lightop.moe import routing

    copy_bw_gbs = measure_copy_bandwidth_gbs(
        warmup=min(args.warmup, 10), iters=min(args.iters, 50)
    )
    if copy_bw_gbs is not None:
        print(f"[bench] Calibrated D2D HBM Copy Bandwidth Ceiling: {copy_bw_gbs:.2f} GB/s")

    rows: list[dict[str, object]] = []

    # 3.1 topk_softmax benchmark
    topk_shapes = [
        # (tokens, experts, topk, dtype, renormalize)
        (1,     256, 8, torch.bfloat16, True),
        (8,     256, 8, torch.bfloat16, True),
        (64,    256, 8, torch.bfloat16, True),
        (256,   256, 8, torch.bfloat16, True),
        (512,   256, 8, torch.bfloat16, True),
        (1024,  256, 8, torch.bfloat16, True),
        (4096,  256, 8, torch.bfloat16, True),
        (16384, 256, 8, torch.bfloat16, True),
        (64,    256, 8, torch.float16,  True),
        (4096,  256, 8, torch.float16,  True),
        (4096,  256, 8, torch.bfloat16, False),
        (64,    256, 8, torch.float32,  True),
        (64,    128, 8, torch.bfloat16, True),
        (4096,  128, 8, torch.bfloat16, True),
        (64,    512, 8, torch.bfloat16, True),
        (4096,  512, 8, torch.bfloat16, True),
        (4096,  512, 8, torch.bfloat16, False),
    ]
    h1 = (
        f"{'topk_softmax (T,E,K)':<22} | {'Dtype/Ren':<10} | "
        f"{'LightOp(us)':<11} | {'TorchRef(us)':<12} | {'Speedup':<8} | {'BW(GB/s)':<9} | {'%CopyBW':<8}"
    )
    print(h1)
    print("-" * len(h1))
    for num_tokens, num_experts, topk, dtype, renorm in topk_shapes:
        gating = torch.randn((num_tokens, num_experts), device="cuda", dtype=dtype)
        weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
        indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

        lightop_native.topk_softmax(weights, indices, None, gating, renorm)
        torch.cuda.synchronize()
        assert_topk_matches_ref(gating, weights, indices, None, renorm, f"bench_{num_tokens}_{num_experts}")

        iters = args.iters if num_tokens <= 4096 else max(40, args.iters // 4)
        lop_us = measure_call_us(
            lambda: lightop_native.topk_softmax(weights, indices, None, gating, renorm),
            args.warmup, iters,
        )
        ref_us = measure_call_us(
            lambda: _torch_topk_softmax_ref(gating, topk, renorm),
            args.warmup, iters,
        )
        logical_bytes = (
            num_tokens * num_experts * gating.element_size()
            + num_tokens * topk * (weights.element_size() + indices.element_size())
        )
        bw_gbs = logical_bytes / (lop_us * 1e-6) / 1e9
        pct_bw = (100.0 * bw_gbs / copy_bw_gbs) if copy_bw_gbs else None
        pct_str = f"{pct_bw:6.1f}%" if pct_bw is not None else "   N/A"
        dt_short = f"{str(dtype).removeprefix('torch.')[:4]}/{'R' if renorm else 'F'}"
        print(
            f"{f'{num_tokens}x{num_experts}x{topk}':<22} | {dt_short:<10} | "
            f"{lop_us:<11.2f} | {ref_us:<12.2f} | {f'{ref_us / lop_us:.2f}x':<8} | "
            f"{bw_gbs:<9.2f} | {pct_str:<8}"
        )
        rows.append(
            {
                "op": "topk_softmax",
                "tokens": num_tokens,
                "experts": num_experts,
                "topk": topk,
                "variant": dt_short,
                "lightop_us": round(lop_us, 3),
                "baseline_us": round(ref_us, 3),
                "speedup_vs_baseline": round(ref_us / lop_us, 3),
                "eff_bw_gbs": round(bw_gbs, 2),
                "pct_copy_bw": round(pct_bw, 2) if pct_bw is not None else "",
                "toks_per_ms": round((num_tokens / (lop_us * 1e-3)) / 1000.0, 2),
            }
        )

    # 3.2 moe_align_block_size_out benchmark (uniform vs skewed vs EP, threshold transitions)
    align_bench_shapes = [
        # (tokens, experts, topk, block_size, is_ep, skewed)
        (16,    256, 8, 16, False, False),
        (64,    256, 8, 16, False, False),
        (256,   256, 8, 16, False, False),
        (512,   256, 8, 16, False, False),
        (512,   256, 8, 16, False, True),
        (1024,  256, 8, 16, False, False),
        (1024,  256, 8, 16, False, True),
        (4096,  256, 8, 16, False, False),
        (4096,  256, 8, 16, False, True),
        (4096,  256, 8, 16, True,  False),
        (16384, 256, 8, 16, False, False),
        (16384, 256, 8, 16, False, True),
        (4096,  600, 8, 16, False, False),
    ]
    h2 = (
        f"\n{'moe_align (T,E,BS)':<22} | {'Mode':<10} | "
        f"{'OutAPI(us)':<11} | {'CompatAPI(us)':<13} | {'VsCompat':<8} | {'BW(GB/s)':<9} | {'Toks/ms':<8}"
    )
    print(h2)
    print("-" * (len(h2) - 1))
    for num_tokens, num_experts, topk, bs, is_ep, skewed in align_bench_shapes:
        if skewed:
            hot = torch.tensor([0, num_experts - 1], device="cuda", dtype=torch.int32)
            topk_ids = hot[torch.randint(0, 2, (num_tokens, topk), device="cuda")]
        else:
            topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device="cuda", dtype=torch.int32)

        emap = None
        if is_ep:
            emap = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
            emap[num_experts - (num_experts // 4) :] = torch.arange(
                num_experts // 4, device="cuda", dtype=torch.int32
            )

        numel = topk_ids.numel()
        max_padded = numel + num_experts * (bs - 1)
        max_blocks = math.ceil(max_padded / bs)
        sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device="cuda")
        expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device="cuda")
        num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")

        routing.moe_align_block_size_out(
            topk_ids, num_experts, bs, sorted_ids, expert_ids, num_tokens_post_pad,
            expert_map=emap, is_ep=is_ep, is_fuse_fill=True,
        )
        torch.cuda.synchronize()
        assert_align_matches_ref(
            topk_ids, num_experts, bs, sorted_ids, expert_ids, num_tokens_post_pad,
            emap, None, None, is_ep, True, f"bench_align_{num_tokens}_{num_experts}",
        )

        iters = args.iters if num_tokens <= 4096 else max(40, args.iters // 4)
        out_us = measure_call_us(
            lambda: routing.moe_align_block_size_out(
                topk_ids, num_experts, bs, sorted_ids, expert_ids, num_tokens_post_pad,
                expert_map=emap, is_ep=is_ep, is_fuse_fill=True,
            ),
            args.warmup, iters,
        )
        compat_us = measure_call_us(
            lambda: routing.moe_align_block_size(
                topk_ids, bs, num_experts, expert_map=emap, ignore_invalid_experts=is_ep
            ),
            args.warmup, iters,
        )
        got_total = int(num_tokens_post_pad.item())
        logical_bytes = (
            numel * topk_ids.element_size()
            + max_padded * 4
            + (got_total // bs) * 4
        )
        bw_gbs = logical_bytes / (out_us * 1e-6) / 1e9
        pct_bw = (100.0 * bw_gbs / copy_bw_gbs) if copy_bw_gbs else None
        mode_tag = ("EP/" if is_ep else "") + ("skew" if skewed else "unif")
        tok_per_ms = (numel / (out_us * 1e-3)) / 1000.0
        print(
            f"{f'{num_tokens}x{num_experts}x{bs}':<22} | {mode_tag:<10} | "
            f"{out_us:<11.2f} | {compat_us:<13.2f} | {f'{compat_us / out_us:.2f}x':<8} | "
            f"{bw_gbs:<9.2f} | {tok_per_ms:<8.1f}"
        )
        rows.append(
            {
                "op": "moe_align_block_size_out",
                "tokens": num_tokens,
                "experts": num_experts,
                "topk": topk,
                "variant": f"bs{bs}_{mode_tag}",
                "lightop_us": round(out_us, 3),
                "baseline_us": round(compat_us, 3),
                "speedup_vs_baseline": round(compat_us / out_us, 3),
                "eff_bw_gbs": round(bw_gbs, 2),
                "pct_copy_bw": round(pct_bw, 2) if pct_bw is not None else "",
                "toks_per_ms": round(tok_per_ms, 2),
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
# Pytest Entrypoints (compatible with `pytest test/test_moe_routing_gfx946.py`)
# =============================================================================

def test_moe_routing_gfx946_isa_audit() -> None:
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


def test_moe_routing_gfx946_correctness() -> None:
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
        description="gfx946 verification, ISA audit & benchmark for topk_softmax and moe_align_block_size_out"
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
    print("\nALL GFX946 MOE ROUTING & ALIGNMENT VERIFICATION STAGES COMPLETED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
