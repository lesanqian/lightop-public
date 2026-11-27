# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import csv
import glob
import os
import sys
import unittest

import torch

# Running this file from the repository makes the source-only ``lightop``
# package shadow the compiled extension. Prefer the current build output when
# it exists, while retaining the normal installed-package path elsewhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_LIB_DIRS = sorted(glob.glob(os.path.join(REPO_ROOT, "build", "lib.*")))
if BUILD_LIB_DIRS:
    sys.path.insert(0, BUILD_LIB_DIRS[-1])

from lightop import op


NUM_EXPERTS = 256
TOPK = 8
DTYPE = torch.float16
TOKEN_SIZES = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def validate_topk_result(
    testcase: unittest.TestCase,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    source_rows: torch.Tensor,
) -> float:
    """Validate top-k values without requiring a specific ID for equal logits."""
    num_tokens = gating_output.size(0)

    testcase.assertTrue(torch.all((topk_indices >= 0) & (topk_indices < NUM_EXPERTS)))
    sorted_indices = torch.sort(topk_indices, dim=-1).values
    if TOPK > 1:
        testcase.assertTrue(torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]))

    selected_logits = torch.gather(
        gating_output.float(), 1, topk_indices.to(torch.int64)
    )
    selected_logits_sorted = torch.sort(
        selected_logits, dim=-1, descending=True
    ).values
    reference_logits = torch.topk(
        gating_output.float(), TOPK, dim=-1, sorted=True
    ).values
    testcase.assertTrue(torch.equal(selected_logits_sorted, reference_logits))

    reference_weights = torch.softmax(selected_logits, dim=-1)
    max_weight_error = (topk_weights - reference_weights).abs().max().item()
    testcase.assertTrue(
        torch.allclose(topk_weights, reference_weights, rtol=2e-5, atol=2e-6),
        msg=f"max weight error: {max_weight_error:.6e}",
    )

    rows = torch.arange(num_tokens, device=gating_output.device)[:, None]
    ranks = torch.arange(TOPK, device=gating_output.device)[None, :]
    expected_source_rows = ranks * num_tokens + rows
    testcase.assertTrue(
        torch.equal(source_rows, expected_source_rows.to(source_rows.dtype))
    )
    return max_weight_error


def benchmark_topk(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    source_rows: torch.Tensor,
    warmup_iters: int,
    test_iters: int,
) -> float:
    for _ in range(warmup_iters):
        op.topk_softmax(
            topk_weights, topk_indices, source_rows, gating_output, True
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(test_iters):
        op.topk_softmax(
            topk_weights, topk_indices, source_rows, gating_output, True
        )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / test_iters


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm device is required")
class TestTopkSoftmax256Fp16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = []
        print("\n=== TopK Softmax: FP16, 256 experts, topk=8, renormalize=True ===")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"{'Tokens':>8} | {'Latency (us)':>12} | {'Max weight error':>16} | Check")
        print("-" * 62)

    @classmethod
    def tearDownClass(cls):
        output_path = os.environ.get(
            "TOPK_BENCH_OUTPUT", "softmax_topk_256_fp16_results.csv"
        )
        if not cls.results:
            return
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=cls.results[0].keys())
            writer.writeheader()
            writer.writerows(cls.results)
        print(f"Results saved to: {os.path.abspath(output_path)}")

    def test_target_sizes(self):
        warmup_iters = int(os.environ.get("TOPK_BENCH_WARMUP", "100"))
        default_iters = int(os.environ.get("TOPK_BENCH_ITERS", "1000"))

        for num_tokens in TOKEN_SIZES:
            with self.subTest(num_tokens=num_tokens):
                torch.manual_seed(2026 + num_tokens)
                gating_output = torch.randn(
                    (num_tokens, NUM_EXPERTS), device="cuda", dtype=DTYPE
                )
                topk_weights = torch.empty(
                    (num_tokens, TOPK), device="cuda", dtype=torch.float32
                )
                topk_indices = torch.empty(
                    (num_tokens, TOPK), device="cuda", dtype=torch.int32
                )
                source_rows = torch.empty_like(topk_indices)

                op.topk_softmax(
                    topk_weights,
                    topk_indices,
                    source_rows,
                    gating_output,
                    True,
                )
                torch.cuda.synchronize()
                max_weight_error = validate_topk_result(
                    self,
                    gating_output,
                    topk_weights,
                    topk_indices,
                    source_rows,
                )

                test_iters = (
                    default_iters
                    if num_tokens <= 4096
                    else max(100, default_iters // 4)
                )
                latency_us = benchmark_topk(
                    gating_output,
                    topk_weights,
                    topk_indices,
                    source_rows,
                    warmup_iters,
                    test_iters,
                )
                print(
                    f"{num_tokens:8d} | {latency_us:12.3f} | "
                    f"{max_weight_error:16.3e} | PASS"
                )
                self.results.append(
                    {
                        "num_tokens": num_tokens,
                        "num_experts": NUM_EXPERTS,
                        "topk": TOPK,
                        "dtype": str(DTYPE),
                        "renormalize": True,
                        "latency_us": f"{latency_us:.6f}",
                        "max_weight_error": f"{max_weight_error:.6e}",
                        "status": "PASS",
                    }
                )


def validate_topk_result_general(
    testcase: unittest.TestCase,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    source_rows: torch.Tensor,
    renormalize: bool,
) -> float:
    """Validate top-k values and weights for general (E, K, dtype, renorm)."""
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    if num_tokens == 0:
        return 0.0

    testcase.assertTrue(torch.all((topk_indices >= 0) & (topk_indices < num_experts)))
    sorted_indices = torch.sort(topk_indices.to(torch.int64), dim=-1).values
    if topk > 1:
        testcase.assertTrue(torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]))

    logits_f32 = gating_output.float()
    selected_logits = torch.gather(logits_f32, 1, topk_indices.to(torch.int64))
    selected_logits_sorted = torch.sort(
        selected_logits, dim=-1, descending=True
    ).values
    reference_logits = torch.topk(logits_f32, topk, dim=-1, sorted=True).values
    testcase.assertTrue(torch.equal(selected_logits_sorted, reference_logits))

    if renormalize:
        reference_weights = torch.softmax(selected_logits, dim=-1)
    else:
        full_probs = torch.softmax(logits_f32, dim=-1)
        reference_weights = torch.gather(
            full_probs, 1, topk_indices.to(torch.int64)
        )

    max_weight_error = (topk_weights - reference_weights).abs().max().item()
    testcase.assertTrue(
        torch.allclose(topk_weights, reference_weights, rtol=1e-4, atol=2e-5),
        msg=(
            f"max weight error: {max_weight_error:.6e} "
            f"(E={num_experts}, K={topk}, dtype={gating_output.dtype}, "
            f"renorm={renormalize})"
        ),
    )

    rows = torch.arange(num_tokens, device=gating_output.device)[:, None]
    ranks = torch.arange(topk, device=gating_output.device)[None, :]
    expected_source_rows = ranks * num_tokens + rows
    testcase.assertTrue(
        torch.equal(source_rows, expected_source_rows.to(source_rows.dtype))
    )
    return max_weight_error


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm device is required")
class TestTopkSoftmaxGfx946Matrix(unittest.TestCase):
    """Coverage matrix for BF16/FP16/FP32 packed16, permlane16/32, and fixes."""

    def test_dtype_expert_topk_matrix(self):
        configs = [
            # (num_tokens, num_experts, topk, dtype, idx_dtype, renormalize)
            (0, 256, 8, torch.bfloat16, torch.int32, True),
            (1, 128, 8, torch.bfloat16, torch.int32, True),
            (64, 128, 8, torch.bfloat16, torch.int64, True),
            (1025, 128, 8, torch.float16, torch.int32, True),
            (7, 256, 8, torch.bfloat16, torch.int32, True),
            (512, 256, 8, torch.bfloat16, torch.int64, True),
            (513, 256, 8, torch.bfloat16, torch.int32, True),
            (1025, 256, 8, torch.bfloat16, torch.int32, True),
            (64, 256, 8, torch.float32, torch.int32, True),
            (64, 256, 8, torch.bfloat16, torch.int32, False),
            (64, 256, 8, torch.float16, torch.int64, False),
            (16, 512, 8, torch.bfloat16, torch.int32, True),
            (1025, 512, 8, torch.bfloat16, torch.int64, True),
            (32, 512, 8, torch.float32, torch.int32, True),
            (32, 512, 8, torch.float32, torch.int32, False),
            (32, 160, 8, torch.bfloat16, torch.int32, True),
            (32, 160, 8, torch.bfloat16, torch.int32, False),
            (64, 64, 4, torch.bfloat16, torch.int32, True),
            (64, 256, 4, torch.float32, torch.int32, False),
            (64, 64, 2, torch.bfloat16, torch.int32, True),
            (64, 256, 2, torch.float16, torch.int64, False),
            (32, 512, 2, torch.float32, torch.int32, True),
        ]
        for num_tokens, num_experts, topk, dtype, idx_dtype, renorm in configs:
            with self.subTest(
                T=num_tokens,
                E=num_experts,
                K=topk,
                dtype=dtype,
                idx_dtype=idx_dtype,
                renorm=renorm,
            ):
                torch.manual_seed(1000 + num_tokens + num_experts + topk)
                gating_output = torch.randn(
                    (num_tokens, num_experts), device="cuda", dtype=dtype
                )
                topk_weights = torch.empty(
                    (num_tokens, topk), device="cuda", dtype=torch.float32
                )
                topk_indices = torch.empty(
                    (num_tokens, topk), device="cuda", dtype=idx_dtype
                )
                source_rows = torch.empty(
                    (num_tokens, topk), device="cuda", dtype=torch.int32
                )
                op.topk_softmax(
                    topk_weights, topk_indices, source_rows, gating_output, renorm
                )
                torch.cuda.synchronize()
                validate_topk_result_general(
                    self,
                    gating_output,
                    topk_weights,
                    topk_indices,
                    source_rows,
                    renorm,
                )

    def test_tie_breaking_and_signed_zero(self):
        for dtype in (torch.bfloat16, torch.float16):
            for num_experts in (128, 256, 512):
                with self.subTest(dtype=dtype, num_experts=num_experts):
                    gating = torch.full(
                        (4, num_experts), -1.0, device="cuda", dtype=dtype
                    )
                    # Equal +0.0 and -0.0 logits must tie-break by smallest expert ID.
                    gating[:, 10] = 0.0
                    gating[:, 5] = -0.0
                    gating[:, 2] = 0.0
                    gating[:, 1] = -0.0
                    gating[:, 0] = 0.0
                    topk_weights = torch.empty(
                        (4, 8), device="cuda", dtype=torch.float32
                    )
                    topk_indices = torch.empty(
                        (4, 8), device="cuda", dtype=torch.int32
                    )
                    source_rows = torch.empty_like(topk_indices)
                    op.topk_softmax(
                        topk_weights, topk_indices, source_rows, gating, True
                    )
                    torch.cuda.synchronize()
                    expected_top5 = torch.tensor(
                        [0, 1, 2, 5, 10], device="cuda", dtype=torch.int32
                    )
                    for row in range(4):
                        self.assertTrue(
                            torch.equal(topk_indices[row, :5], expected_top5),
                            msg=f"got {topk_indices[row, :5].tolist()}",
                        )


if __name__ == "__main__":
    unittest.main()
