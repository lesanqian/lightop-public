# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Migration test for LMSlim blockwise hipBLASLt GEMM."""

import inspect
import math

import pytest
import torch

from lightop._lmslim_native.quantize.quant_ops import (
    hipblaslt_w8a8_blockwise_gemm as native_api,
)
from lightop.gemm_ops import BlockSize, hipblaslt_w8a8_blockwise_gemm


def test_hipblaslt_w8a8_blockwise_gemm_export() -> None:
    assert hipblaslt_w8a8_blockwise_gemm is native_api
    assert inspect.signature(hipblaslt_w8a8_blockwise_gemm) == inspect.signature(native_api)


def test_hipblaslt_w8a8_blockwise_gemm_matches_torch_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    if device.gcnArchName.split(":")[0] == "gfx936":
        pytest.skip("blockwise FP8 GEMM uses the PyTorch fallback on gfx936")
    if device.name == "BW151":
        pytest.skip("BW151 hipBLASLt has no blockwise FP8 algorithm for this test")

    torch.manual_seed(10)
    m, n, k = 256, 128, 512
    block_size = 128
    a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((k, n), device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.rand(
        (m, math.ceil(k / block_size)), device="cuda", dtype=torch.float32
    ) * 0.1
    scale_b = torch.rand(
        (math.ceil(k / block_size), math.ceil(n / block_size)),
        device="cuda",
        dtype=torch.float32,
    ) * 0.1

    status, actual = hipblaslt_w8a8_blockwise_gemm(
        a,
        b,
        scale_a,
        scale_b,
        m,
        n,
        k,
        "NN",
        torch.bfloat16,
        BlockSize.block_128x128,
    )

    k_blocks = torch.arange(k, device="cuda") // block_size
    n_blocks = torch.arange(n, device="cuda") // block_size
    expected = (
        a.float() * scale_a[:, k_blocks]
    ) @ (
        b.float() * scale_b[k_blocks][:, n_blocks]
    )
    torch.cuda.synchronize()

    assert status is True
    torch.testing.assert_close(actual.view(m, n), expected.to(torch.bfloat16), rtol=0, atol=1)


@pytest.mark.parametrize("transpose_flag", ["NN", "NT"])
@pytest.mark.parametrize("block_size", [BlockSize.block_128x128, BlockSize.block_64x64])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("m,n,k", [(16, 576, 1152), (32, 2112, 128), (64, 256, 512)])
def test_hipblaslt_w8a8_blockwise_gemm_gfx946_ue8m0_dispatch(
    monkeypatch, transpose_flag, block_size, batch, m, n, k
) -> None:
    from types import SimpleNamespace
    import lightop._lmslim_native.quantize.quant_ops as quant_ops_mod

    props = SimpleNamespace(gcnArchName="gfx946:sramecc+")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)

    captured = {}

    def fake_native(a, b, sa, sb, out_dtype, mm, nn, kk, bb, tf, blk, bias):
        captured.update(
            m=mm, n=nn, k=kk, batch=bb, transpose_flag=tf, block_size=blk, has_bias=bias is not None
        )
        return torch.zeros((bb, mm, nn), dtype=out_dtype)

    monkeypatch.setattr(
        quant_ops_mod.lightop_native,
        "fp8_blockwise_gemm_ue8m0_gfx946",
        fake_native,
        raising=False,
    )

    blk = 128 if block_size == BlockSize.block_128x128 else 64
    k_blks = math.ceil(k / blk)
    n_blks = math.ceil(n / blk)
    a_shape = (batch, m, k) if batch > 1 else (m, k)
    b_shape = ((batch, k, n) if batch > 1 else (k, n)) if transpose_flag == "NN" else (
        (batch, n, k) if batch > 1 else (n, k)
    )
    sa_shape = (batch, m, k_blks) if batch > 1 else (m, k_blks)
    sb_shape = (
        ((batch, k_blks, n_blks) if batch > 1 else (k_blks, n_blks))
        if transpose_flag == "NN"
        else ((batch, n_blks, k_blks) if batch > 1 else (n_blks, k_blks))
    )
    a = torch.zeros(a_shape, dtype=torch.float8_e4m3fn)
    b = torch.zeros(b_shape, dtype=torch.float8_e4m3fn)
    sa = torch.ones(sa_shape, dtype=torch.float32)
    sb = torch.ones(sb_shape, dtype=torch.float32)

    ok, out = hipblaslt_w8a8_blockwise_gemm(
        a, b, sa, sb, m, n, k, transpose_flag, torch.bfloat16, block_size, use_ue8m0=True
    )
    assert ok is True
    assert out.shape == (batch, m, n)
    assert captured == {
        "m": m,
        "n": n,
        "k": k,
        "batch": batch,
        "transpose_flag": transpose_flag,
        "block_size": blk,
        "has_bias": False,
    }


def test_hipblaslt_w8a8_blockwise_gemm_gfx946_unaligned_and_non_ue8m0_fallback(monkeypatch) -> None:
    from types import SimpleNamespace
    import lightop._lmslim_native.quantize.quant_ops as quant_ops_mod

    props = SimpleNamespace(gcnArchName="gfx946")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)

    def should_not_call(*args, **kwargs):
        raise AssertionError("native gfx946 UE8M0 kernel should not be called for fallback cases")

    monkeypatch.setattr(
        quant_ops_mod.lightop_native,
        "fp8_blockwise_gemm_ue8m0_gfx946",
        should_not_call,
        raising=False,
    )

    # Case 1: N not 16-byte aligned (e.g., n=20, k=128) -> safe fallback even with use_ue8m0=True
    a = torch.ones((4, 128), dtype=torch.float8_e4m3fn)
    b_nt = torch.ones((20, 128), dtype=torch.float8_e4m3fn)
    sa = torch.ones((4, 1), dtype=torch.float32)
    sb = torch.ones((1, 1), dtype=torch.float32)
    ok, out = hipblaslt_w8a8_blockwise_gemm(
        a, b_nt, sa, sb, 4, 20, 128, "NT", torch.bfloat16, BlockSize.block_128x128, use_ue8m0=True
    )
    assert ok is True
    assert out.shape == (1, 4, 20)
    torch.testing.assert_close(out, torch.full((1, 4, 20), 128.0, dtype=torch.bfloat16))

    # Case 2: use_ue8m0=False -> safe fallback even when aligned
    b_nn = torch.ones((128, 32), dtype=torch.float8_e4m3fn)
    ok2, out2 = hipblaslt_w8a8_blockwise_gemm(
        a, b_nn, sa, sb, 4, 32, 128, "NN", torch.bfloat16, BlockSize.block_128x128, use_ue8m0=False
    )
    assert ok2 is True
    assert out2.shape == (1, 4, 32)
    torch.testing.assert_close(out2, torch.full((1, 4, 32), 128.0, dtype=torch.bfloat16))
