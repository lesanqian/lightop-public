# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for FP32 residual sums in fused add + RMSNorm."""

import unittest

import torch

from lightop import op


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm device required")
class TestFusedAddRMSNormFP32(unittest.TestCase):
    def check_case(self, rows, hidden, dtype, *, gemma=False, limit=-1.0,
                   offset=0, cancellation=False):
        torch.manual_seed(1517 + hidden)
        # An offset of one element exercises the unaligned fallback.
        x = torch.randn(rows * hidden + offset, device="cuda", dtype=dtype)[offset:]
        x = x.view(rows, hidden)
        r = torch.randn(rows * hidden + offset, device="cuda", dtype=dtype)[offset:]
        r = r.view(rows, hidden)
        if cancellation:
            r.copy_((-x.float() + 0.003).to(dtype))
        w = torch.randn(hidden + offset, device="cuda", dtype=dtype)[offset:]
        x0, r0, w0 = x.clone(), r.clone(), w.clone()
        summed = x0.float() + r0.float()
        reference = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-5)
        reference = reference * (w0.float() + (1.0 if gemma else 0.0))
        if limit >= 0:
            reference = reference.clamp(-limit, limit)
        expected = reference.to(dtype)
        ptrs = (x.data_ptr(), r.data_ptr())
        if gemma:
            op.gemma_fused_add_rmsnorm(x, r, w, 1e-5)
        else:
            op.fused_add_rms_norm_opt(x, r, w, 1e-5, limit)
        self.assertEqual((x.data_ptr(), r.data_ptr()), ptrs)
        torch.testing.assert_close(w, w0, rtol=0, atol=0)
        torch.testing.assert_close(r, summed.to(dtype), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(x).all())
        error = ((x.float() - expected.float()).norm()
                 / expected.float().norm().clamp_min(1e-30)).item()
        # Allow final-cast boundary differences, but reject premature sum rounding.
        tolerance = 1e-4 if hidden in (4096, 8192) else 5e-4
        self.assertLess(error, tolerance, (rows, hidden, dtype, gemma, limit, error))
        if limit >= 0:
            self.assertLessEqual(x.float().abs().max().item(), limit)

    def check_no_residual_case(self, rows, hidden, dtype, *, gemma=False,
                               limit=-1.0, offset=0, out_offset=0,
                               inplace=False):
        torch.manual_seed(2026 + hidden)
        x = torch.randn(rows * hidden + offset, device="cuda", dtype=dtype)[offset:]
        x = x.view(rows, hidden)
        w = torch.randn(hidden + offset, device="cuda", dtype=dtype)[offset:]
        x0, w0 = x.clone(), w.clone()
        x_fp32 = x0.float()
        reference = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + 1e-5)
        reference = reference * (w0.float() + (1.0 if gemma else 0.0))
        if limit >= 0:
            reference = reference.clamp(-limit, limit)
        expected = reference.to(dtype)

        if inplace:
            out = x
        else:
            out = torch.empty(rows * hidden + out_offset, device="cuda", dtype=dtype)[out_offset:]
            out = out.view(rows, hidden)
        out_ptr = out.data_ptr()
        if gemma:
            op.gemma_rmsnorm(out, x, w, 1e-5)
        else:
            op.rms_norm_opt(out, x, w, 1e-5, limit)
        self.assertEqual(out.data_ptr(), out_ptr)
        torch.testing.assert_close(w, w0, rtol=0, atol=0)
        if not inplace:
            torch.testing.assert_close(x, x0, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(out).all())
        error = ((out.float() - expected.float()).norm()
                 / expected.float().norm().clamp_min(1e-30)).item()
        tolerance = 1e-4 if hidden in (4096, 8192) else 5e-4
        self.assertLess(error, tolerance, (rows, hidden, dtype, gemma, limit, error))
        if limit >= 0:
            self.assertLessEqual(out.float().abs().max().item(), limit)

    def test_dispatch_shapes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (64, 128, 256, 512, 768, 1024, 1536, 2048,
                           2560, 3584, 4096, 5376, 8192, 16384, 65, 520, 16392):
                with self.subTest(dtype=dtype, hidden=hidden):
                    self.check_case(17, hidden, dtype)

    def test_clamp_and_gemma(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (64, 128, 256, 512, 2560, 3584, 4096, 5376, 8192, 65, 520, 16392):
                for mode in ("clamp", "gemma"):
                    with self.subTest(dtype=dtype, hidden=hidden, mode=mode):
                        self.check_case(17, hidden, dtype,
                                        gemma=mode == "gemma",
                                        limit=0.5 if mode == "clamp" else -1.0)

    def test_no_residual_rms_norm_and_gemma(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (64, 128, 256, 512, 1024, 2048, 2560, 3584,
                           4096, 5376, 8192, 16384, 65, 520, 16392):
                for gemma in (False, True):
                    with self.subTest(dtype=dtype, hidden=hidden, gemma=gemma):
                        self.check_no_residual_case(17, hidden, dtype, gemma=gemma)
        for dtype in (torch.bfloat16, torch.float16):
            for hidden in (128, 512, 4096):
                with self.subTest(dtype=dtype, hidden=hidden, mode="clamp"):
                    self.check_no_residual_case(17, hidden, dtype, gemma=False, limit=0.5)
                with self.subTest(dtype=dtype, hidden=hidden, mode="inplace"):
                    self.check_no_residual_case(17, hidden, dtype, gemma=True, inplace=True)
                with self.subTest(dtype=dtype, hidden=hidden, mode="unaligned_out"):
                    self.check_no_residual_case(17, hidden, dtype, gemma=True, out_offset=1)
                with self.subTest(dtype=dtype, hidden=hidden, mode="grid288"):
                    self.check_no_residual_case(288, hidden, dtype, gemma=True)

    def test_unaligned_cancellation_and_grid_reuse(self):
        for dtype in (torch.bfloat16, torch.float16):
            for hidden in (65, 128, 512, 4096, 8192):
                with self.subTest(dtype=dtype, hidden=hidden, unaligned=True):
                    self.check_case(17, hidden, dtype, offset=1)
                with self.subTest(dtype=dtype, hidden=hidden, cancellation=True):
                    self.check_case(17, hidden, dtype, cancellation=True)
                with self.subTest(dtype=dtype, hidden=hidden, gemma_cancel=True):
                    self.check_case(17, hidden, dtype, gemma=True, cancellation=True)
            # Cover both the general grid-reuse path and specialized tiled paths.
            for _ in range(3):
                self.check_case(288, 2048, dtype)
                self.check_case(288, 4096, dtype)
                self.check_case(288, 8192, dtype)
                self.check_case(288, 4096, dtype, gemma=True)

    def test_graph_replay(self):
        for hidden in (128, 4096, 8192):
            for gemma in (False, True):
                dtype = torch.bfloat16
                torch.manual_seed(42)
                x = torch.randn(17, hidden, device="cuda", dtype=dtype)
                r = torch.randn_like(x)
                w = torch.randn(hidden, device="cuda", dtype=dtype)
                x0, r0 = x.clone(), r.clone()
                if gemma:
                    op.gemma_fused_add_rmsnorm(x, r, w, 1e-5)
                else:
                    op.fused_add_rms_norm_opt(x, r, w, 1e-5, -1.0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    x.copy_(x0)
                    r.copy_(r0)
                    if gemma:
                        op.gemma_fused_add_rmsnorm(x, r, w, 1e-5)
                    else:
                        op.fused_add_rms_norm_opt(x, r, w, 1e-5, -1.0)
                for _ in range(3):
                    x0.normal_()
                    r0.normal_()
                    graph.replay()
                    summed = x0.float() + r0.float()
                    wt = w.float() + (1.0 if gemma else 0.0)
                    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True)
                                                    + 1e-5) * wt).to(dtype)
                    error = ((x.float() - expected.float()).norm()
                             / expected.float().norm()).item()
                    tol = 1e-4 if hidden in (4096, 8192) else 5e-4
                    self.assertLess(error, tol)
                    torch.testing.assert_close(r, summed.to(dtype), rtol=0, atol=0)

    def test_empty(self):
        for hidden in (65, 128, 512, 4096, 8192):
            x = torch.empty(0, hidden, device="cuda", dtype=torch.bfloat16)
            r = torch.empty_like(x)
            out = torch.empty_like(x)
            w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
            op.fused_add_rms_norm_opt(x, r, w, 1e-5, -1.0)
            op.gemma_fused_add_rmsnorm(x, r, w, 1e-5)
            op.rms_norm_opt(out, x, w, 1e-5, -1.0)
            op.gemma_rmsnorm(out, x, w, 1e-5)
            self.assertEqual(x.shape, (0, hidden))
            self.assertEqual(out.shape, (0, hidden))
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
