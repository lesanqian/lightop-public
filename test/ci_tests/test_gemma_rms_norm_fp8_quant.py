# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Gemma/standard RMSNorm + per-token FP8 quantization."""

import unittest
from typing import Optional

import torch

from lightop.rmsnorm import (
    gemma_rms_norm_per_token_fp8_quant,
    rms_norm_per_token_fp8_quant,
)


def ref_rms_norm_per_token_fp8_quant(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-5,
    fp8type: int = 0,
    residual: Optional[torch.Tensor] = None,
    gemma: bool = True,
):
    """Independent PyTorch FP32 reference matching the FP32-residual-sum contract."""
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
    scales_flat = scales.reshape(-1, 1)
    return q, scales_flat, elem, ref_res


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm device required")
class TestGemmaRMSNormFP8Quant(unittest.TestCase):
    def check_case(
        self,
        rows: int,
        hidden: int,
        dtype: torch.dtype,
        *,
        gemma: bool = True,
        fp8type: int = 0,
        with_residual: bool = False,
        update_input: bool = False,
        offset: int = 0,
        pad: int = 0,
        cancellation: bool = False,
        zero_input: bool = False,
    ):
        torch.manual_seed(9460 + hidden + (100 if gemma else 0) + fp8type)
        if pad > 0:
            x_buf = torch.randn(rows, hidden + pad, device="cuda", dtype=dtype)
            x = x_buf[:, :hidden]
        else:
            x_buf = torch.randn(rows * hidden + offset, device="cuda", dtype=dtype)
            x = x_buf[offset:].view(rows, hidden)

        if zero_input:
            x.zero_()

        if with_residual:
            if pad > 0:
                r_buf = torch.randn(rows, hidden + pad, device="cuda", dtype=dtype)
                r = r_buf[:, :hidden]
            else:
                r_buf = torch.randn(rows * hidden + offset, device="cuda", dtype=dtype)
                r = r_buf[offset:].view(rows, hidden)
            if zero_input:
                r.zero_()
            elif cancellation:
                r.copy_((-x.float() + 0.003).to(dtype))
        else:
            r = None

        w = torch.randn(hidden + offset, device="cuda", dtype=dtype)[offset:]
        x0 = x.clone()
        r0 = r.clone() if r is not None else None
        w0 = w.clone()

        ref_q, ref_scales, ref_elem, ref_res = ref_rms_norm_per_token_fp8_quant(
            x0, w0, eps=1e-5, fp8type=fp8type, residual=r0, gemma=gemma
        )

        fn = (
            gemma_rms_norm_per_token_fp8_quant
            if gemma
            else rms_norm_per_token_fp8_quant
        )
        out_q, out_scales = fn(
            x, w, epsilon=1e-5, fp8type=fp8type, residual=r, update_input=update_input
        )

        expected_fp8_dtype = torch.float8_e4m3fn if fp8type == 0 else torch.float8_e5m2
        self.assertEqual(out_q.dtype, expected_fp8_dtype)
        self.assertEqual(out_q.shape, (rows, hidden))
        self.assertEqual(out_scales.shape, (rows, 1))
        self.assertEqual(out_scales.dtype, torch.float32)

        torch.testing.assert_close(w, w0, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(out_scales).all())
        self.assertTrue(torch.isfinite(out_q.float()).all())

        if with_residual:
            # RNE packed writeback must match PyTorch FP32 sum -> dtype bit-for-bit.
            torch.testing.assert_close(r, ref_res, rtol=0, atol=0)

        if update_input:
            elem_err = (
                (x.float() - ref_elem.float()).norm()
                / ref_elem.float().norm().clamp_min(1e-30)
            ).item()
            self.assertLess(elem_err, 5e-3, (rows, hidden, dtype, gemma, elem_err))
        else:
            torch.testing.assert_close(x, x0, rtol=0, atol=0)

        if zero_input:
            fp8_max = 448.0 if fp8type == 0 else 57344.0
            min_scale = 1.0 / (fp8_max * 512.0)
            torch.testing.assert_close(
                out_scales,
                torch.full_like(out_scales, min_scale),
                rtol=1e-6,
                atol=1e-12,
            )
            torch.testing.assert_close(
                out_q.float(), torch.zeros_like(out_q.float()), rtol=0, atol=0
            )
            return

        # Scale check (tight tolerance; allows 1 BF16/FP16 ULP boundary on elem absmax).
        scale_rtol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(out_scales, ref_scales, rtol=scale_rtol, atol=1e-6)

        # Dequantized FP8 output comparison against independent PyTorch FP8 reference.
        deq_actual = out_q.float() * out_scales
        deq_ref = ref_q.float() * ref_scales
        rel_err = (
            (deq_actual - deq_ref).norm() / deq_ref.norm().clamp_min(1e-30)
        ).item()
        # E5M2 has 2-bit mantissa (~25% step); E4M3 has 3-bit mantissa (~12.5% step).
        tol = 5e-2 if fp8type == 0 else 1.2e-1
        self.assertLess(
            rel_err,
            tol,
            (rows, hidden, dtype, gemma, fp8type, with_residual, cancellation, rel_err),
        )

    def test_dispatch_shapes_no_residual(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (
                64, 128, 256, 512, 1024, 2048, 2560, 3584,
                4096, 5376, 8192, 16384, 65, 520, 16392,
            ):
                for fp8type in (0, 1):
                    for gemma in (True, False):
                        with self.subTest(
                            dtype=dtype, hidden=hidden, fp8type=fp8type, gemma=gemma
                        ):
                            self.check_case(
                                17, hidden, dtype, gemma=gemma, fp8type=fp8type
                            )

    def test_residual_fp32_sum_and_cancellation(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (
                64, 128, 512, 1024, 2048, 3584, 4096, 8192, 16384, 65, 520, 16392
            ):
                for fp8type in (0, 1):
                    for gemma in (True, False):
                        with self.subTest(
                            dtype=dtype, hidden=hidden, fp8type=fp8type, gemma=gemma
                        ):
                            self.check_case(
                                17,
                                hidden,
                                dtype,
                                gemma=gemma,
                                fp8type=fp8type,
                                with_residual=True,
                            )
        # Catastrophic cancellation: fails if residual sum is prematurely rounded to BF16/FP16.
        for dtype in (torch.bfloat16, torch.float16):
            for hidden in (128, 512, 1024, 4096, 8192, 520):
                for gemma in (True, False):
                    with self.subTest(
                        dtype=dtype, hidden=hidden, gemma=gemma, cancellation=True
                    ):
                        self.check_case(
                            17,
                            hidden,
                            dtype,
                            gemma=gemma,
                            fp8type=0,
                            with_residual=True,
                            cancellation=True,
                        )

    def test_update_input_flag(self):
        for dtype in (torch.bfloat16, torch.float16):
            for hidden in (128, 1024, 4096, 520):
                for with_residual in (False, True):
                    for update_input in (False, True):
                        with self.subTest(
                            dtype=dtype,
                            hidden=hidden,
                            with_residual=with_residual,
                            update_input=update_input,
                        ):
                            self.check_case(
                                17,
                                hidden,
                                dtype,
                                gemma=True,
                                fp8type=0,
                                with_residual=with_residual,
                                update_input=update_input,
                            )

    def test_non_contiguous_unaligned_and_ndim(self):
        # pad=20 exercises loop fallback (previously triggered VMFault on out stride);
        # pad=16 exercises 16B-aligned non-contiguous stride on fast path.
        for dtype in (torch.bfloat16, torch.float16):
            for pad in (16, 20):
                for with_residual in (False, True):
                    with self.subTest(
                        dtype=dtype, pad=pad, with_residual=with_residual
                    ):
                        self.check_case(
                            17,
                            1024,
                            dtype,
                            gemma=True,
                            fp8type=0,
                            with_residual=with_residual,
                            pad=pad,
                            update_input=True,
                        )
            for hidden in (512, 4096):
                for with_residual in (False, True):
                    with self.subTest(
                        dtype=dtype, hidden=hidden, unaligned=True, res=with_residual
                    ):
                        self.check_case(
                            17,
                            hidden,
                            dtype,
                            gemma=True,
                            fp8type=0,
                            with_residual=with_residual,
                            offset=1,
                        )

        # 1D [H] and 3D [B, S, H] shapes.
        for shape in ((1024,), (3, 5, 1024), (2, 7, 4096)):
            torch.manual_seed(77)
            x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
            ref_q, ref_s, _, ref_r = ref_rms_norm_per_token_fp8_quant(
                x.clone(), w.clone(), eps=1e-5, fp8type=0, residual=r.clone(), gemma=True
            )
            out_q, out_s = gemma_rms_norm_per_token_fp8_quant(
                x, w, epsilon=1e-5, fp8type=0, residual=r, update_input=False
            )
            self.assertEqual(out_q.shape, shape)
            torch.testing.assert_close(r, ref_r, rtol=0, atol=0)
            torch.testing.assert_close(out_s, ref_s, rtol=1e-2, atol=1e-6)
            deq_act = out_q.reshape(-1, shape[-1]).float() * out_s
            deq_ref = ref_q.reshape(-1, shape[-1]).float() * ref_s
            rel_err = ((deq_act - deq_ref).norm() / deq_ref.norm().clamp_min(1e-30)).item()
            self.assertLess(rel_err, 5e-2)

    def test_zero_row_empty_and_grid_reuse(self):
        for dtype in (torch.bfloat16, torch.float16):
            for fp8type in (0, 1):
                for with_residual in (False, True):
                    with self.subTest(
                        dtype=dtype, fp8type=fp8type, with_residual=with_residual
                    ):
                        self.check_case(
                            8,
                            1024,
                            dtype,
                            gemma=True,
                            fp8type=fp8type,
                            with_residual=with_residual,
                            zero_input=True,
                        )

        # num_tokens=140 exercises legacy <=128-CU grid_opt=128 reuse and full launch on 192-CU.
        for hidden in (512, 4096, 8192):
            for with_residual in (False, True):
                with self.subTest(hidden=hidden, with_residual=with_residual):
                    self.check_case(
                        140,
                        hidden,
                        torch.bfloat16,
                        gemma=True,
                        fp8type=0,
                        with_residual=with_residual,
                    )

        # Empty [0, hidden] tensors.
        for hidden in (65, 512, 4096, 8192):
            x = torch.empty(0, hidden, device="cuda", dtype=torch.bfloat16)
            r = torch.empty_like(x)
            w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
            out_q, out_s = gemma_rms_norm_per_token_fp8_quant(
                x, w, epsilon=1e-5, fp8type=0, residual=r
            )
            self.assertEqual(out_q.shape, (0, hidden))
            self.assertEqual(out_s.shape, (0, 1))
        torch.cuda.synchronize()

    def test_graph_replay(self):
        for hidden in (1024, 4096):
            for with_residual in (False, True):
                dtype = torch.bfloat16
                torch.manual_seed(946)
                x = torch.randn(17, hidden, device="cuda", dtype=dtype)
                r = torch.randn_like(x) if with_residual else None
                w = torch.randn(hidden, device="cuda", dtype=dtype)
                x0 = x.clone()
                r0 = r.clone() if r is not None else None

                # Warmup
                gemma_rms_norm_per_token_fp8_quant(
                    x, w, epsilon=1e-5, fp8type=0, residual=r, update_input=False
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    x.copy_(x0)
                    if r is not None:
                        r.copy_(r0)
                    out_q, out_s = gemma_rms_norm_per_token_fp8_quant(
                        x, w, epsilon=1e-5, fp8type=0, residual=r, update_input=False
                    )

                for _ in range(3):
                    x0.normal_()
                    if r0 is not None:
                        r0.normal_()
                    graph.replay()
                    ref_q, ref_s, _, ref_r = ref_rms_norm_per_token_fp8_quant(
                        x0, w, eps=1e-5, fp8type=0, residual=r0, gemma=True
                    )
                    if r is not None:
                        torch.testing.assert_close(r, ref_r, rtol=0, atol=0)
                    torch.testing.assert_close(out_s, ref_s, rtol=1e-2, atol=1e-6)
                    deq_act = out_q.float() * out_s
                    deq_ref = ref_q.float() * ref_s
                    rel_err = (
                        (deq_act - deq_ref).norm() / deq_ref.norm().clamp_min(1e-30)
                    ).item()
                    self.assertLess(rel_err, 5e-2)


if __name__ == "__main__":
    unittest.main()
