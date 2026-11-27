# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the categorized preallocated MoE alignment API."""

import math
from unittest.mock import patch

import pytest
import torch

from lightop.moe import routing


def test_moe_align_block_size_out_forwards_ep_and_fill_flags() -> None:
    arguments = [object() for _ in range(9)]
    expected = object()

    with patch.object(
        routing.op,
        "moe_align_block_size",
        return_value=expected,
    ) as native_api:
        result = routing.moe_align_block_size_out(
            *arguments,
            is_ep=True,
            is_fuse_fill=False,
        )

    assert result is expected
    native_api.assert_called_once_with(*arguments, True, False)


def _reference_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    expert_map: torch.Tensor | None = None,
    num_local_tokens: int | None = None,
    is_ep: bool = False,
) -> tuple[list[list[int]], list[int], int, list[bool] | None]:
    """Reference matching LightOp's logical-expert ordering and EP masking."""
    flat = topk_ids.reshape(-1).cpu().tolist()
    topk = topk_ids.size(-1) if topk_ids.dim() >= 2 else 8
    eff_numel = len(flat) if num_local_tokens is None else num_local_tokens * topk
    emap = expert_map.cpu().tolist() if expert_map is not None else None

    if num_local_tokens == 0:
        return [], [], 0, None

    buckets: list[list[int]] = [[] for _ in range(num_experts)]
    mask: list[bool] = []
    for idx in range(eff_numel):
        eid = flat[idx]
        valid = (emap is None) or (emap[eid] != -1)
        mask.append(valid)
        if not is_ep or valid:
            buckets[eid].append(idx)

    expected_blocks: list[list[int]] = []
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
            expected_blocks.append(sorted(buckets[eid]))
        total_post_pad += padded
    return expected_blocks, expected_expert_ids, total_post_pad, mask


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm device is required"
)
@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,block_size,dtype,is_ep,is_fuse_fill,skewed",
    [
        # Small E <= 64 (previously fell into broken small_batch_expert_mode)
        (8, 2, 32, 16, torch.int32, False, True, False),
        (16, 4, 64, 16, torch.int64, True, True, False),
        # Medium size (previously hit CUB_BLOCK_THREADS != threads mismatch)
        (256, 8, 128, 16, torch.int32, False, True, False),
        (256, 8, 256, 32, torch.int32, True, True, True),
        (600, 8, 256, 16, torch.int64, True, True, True),
        # Multi-block threshold & large size (4-XCD fence + EP index fix)
        (512, 8, 256, 64, torch.int32, True, True, False),
        (2560, 8, 256, 16, torch.int32, False, True, True),
        (2560, 8, 256, 16, torch.int64, True, True, False),
        (2560, 8, 256, 24, torch.int32, True, False, False),
        # 512 < E <= 1024 in multi-block mode (previously dropped experts >= 512)
        (2560, 8, 600, 16, torch.int32, False, True, False),
        (2560, 8, 600, 16, torch.int32, True, True, True),
    ],
)
def test_moe_align_block_size_out_gpu_matrix(
    num_tokens: int,
    topk: int,
    num_experts: int,
    block_size: int,
    dtype: torch.dtype,
    is_ep: bool,
    is_fuse_fill: bool,
    skewed: bool,
) -> None:
    torch.manual_seed(42 + num_tokens + num_experts)
    device = "cuda"
    if skewed:
        hot = torch.tensor([0, num_experts - 1], device=device, dtype=dtype)
        topk_ids = hot[
            torch.randint(0, 2, (num_tokens, topk), device=device)
        ]
    else:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), device=device, dtype=dtype
        )

    expert_map = None
    if is_ep:
        # Non-identity EP shard mapping: only upper quarter of experts are local.
        expert_map = torch.full(
            (num_experts,), -1, device=device, dtype=torch.int32
        )
        local_start = (num_experts * 3) // 4
        local_count = num_experts - local_start
        expert_map[local_start:] = torch.arange(
            local_count, device=device, dtype=torch.int32
        )

    numel = topk_ids.numel()
    # Intentionally add +3 tail elements to verify int4 fill tail handling.
    max_padded = numel + num_experts * (block_size - 1) + 3
    max_blocks = math.ceil(max_padded / block_size)

    sorted_ids = (
        torch.empty((max_padded,), dtype=torch.int32, device=device)
        if is_fuse_fill
        else torch.full((max_padded,), numel, dtype=torch.int32, device=device)
    )
    expert_ids = torch.full((max_blocks,), -99, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.full((1,), -1, dtype=torch.int32, device=device)
    expert_mask = torch.empty((numel,), dtype=torch.bool, device=device)

    routing.moe_align_block_size_out(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map=expert_map,
        expert_mask=expert_mask,
        is_ep=is_ep,
        is_fuse_fill=is_fuse_fill,
    )
    torch.cuda.synchronize()

    ref_buckets, ref_eids, ref_total, ref_mask = _reference_moe_align_block_size(
        topk_ids, num_experts, block_size, expert_map=expert_map, is_ep=is_ep
    )

    got_total = int(num_tokens_post_pad.item())
    assert got_total == ref_total
    assert expert_ids[: got_total // block_size].cpu().tolist() == ref_eids
    assert expert_mask.cpu().tolist() == ref_mask

    sorted_cpu = sorted_ids.cpu().tolist()
    if is_fuse_fill:
        # Tail beyond valid output must still hold fill sentinel numel.
        assert all(v == numel for v in sorted_cpu[got_total:])

    # Verify each non-empty expert's contiguous padded segment holds exact token IDs
    # followed by sentinel padding `numel`.
    offset = 0
    for expected_tokens in ref_buckets:
        padded = math.ceil(len(expected_tokens) / block_size) * block_size
        segment = sorted_cpu[offset : offset + padded]
        actual_tokens = sorted(v for v in segment if v != numel)
        pad_count = sum(1 for v in segment if v == numel)
        assert actual_tokens == expected_tokens
        assert pad_count == padded - len(expected_tokens)
        offset += padded
    assert offset == got_total


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm device is required"
)
def test_moe_align_block_size_out_num_local_tokens_non8_and_zero() -> None:
    device = "cuda"
    num_experts, topk, block_size = 128, 4, 16
    topk_ids = torch.randint(
        0, num_experts, (100, topk), device=device, dtype=torch.int32
    )
    numel = topk_ids.numel()
    max_padded = numel + num_experts * (block_size - 1)
    max_blocks = math.ceil(max_padded / block_size)

    for local_tok in (0, 37):
        sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
        expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=device)
        num_tokens_post_pad = torch.full((1,), -1, dtype=torch.int32, device=device)
        num_local_tokens = torch.tensor(
            [local_tok], dtype=torch.int32, device=device
        )

        routing.moe_align_block_size_out(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            num_local_tokens=num_local_tokens,
            is_ep=False,
            is_fuse_fill=True,
        )
        torch.cuda.synchronize()

        _, ref_eids, ref_total, _ = _reference_moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            num_local_tokens=local_tok,
            is_ep=False,
        )
        assert int(num_tokens_post_pad.item()) == ref_total
        if ref_total > 0:
            assert expert_ids[: ref_total // block_size].cpu().tolist() == ref_eids
