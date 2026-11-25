// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for the MQA logits family in full and selective builds.
// Sources are declared in python/lightop/module_registry.json.
// Helpers implemented outside this family (such as dsv41_pair_partners) remain
// in lightop._C and are exposed through the same lightop.op facade.

#include <torch/extension.h>

#include <optional>
#include <vector>

// Same family headers as csrc/pybind/export.cpp's main branch. Paths are
// relative to this file's directory (csrc/mqa_logits/), matching the style of
// export.cpp; the 1118 tree regrouped the family under mqa_logits/{fp8_fp4,
// paged,sparse}/ and moved index_k_int8_dequant under csrc/kernels/.
#include "mqa_logits_api.h"
#include "fp8_fp4/fp8_fp4_mqa_logits.h"
#include "fp8_fp4/dsv41_mqa_prep.h"
#include "paged/paged_mqa_logits.h"
#include "paged/paged_mqa_logits_persisten.h"
#include "paged/paged_mqa_logits_fp4.h"
#include "sparse/bind_sparse_mqa.h"
#include "sparse/sparse_page_mqa.h"
#include "../kernels/index_k_int8_dequant/index_k_int8_dequant.h"

namespace at::native {
// Declaration mirrored verbatim from csrc/mqa_logits/mqa_logits_asm.cu.
torch::Tensor mqa_logits(
    torch::Tensor& Q,
    torch::Tensor& K,
    torch::Tensor& Weights,
    torch::Tensor& KS,
    torch::Tensor& KE,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int head_dim,
    const torch::optional<torch::Tensor> &KV_scale =  torch::nullopt,
    bool clean_logits = true,
    const torch::optional<torch::Tensor> &D_out = torch::nullopt);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp MQA logits operators";
  // Marker proving the aggregator (lightop/core.py) routed to this module.
  m.attr("LIGHTOP_MQA_MODULE") = 1;

  m.def("mqa_logits_topk_sampled_int8", &lightop::mqa_int8::mqa_logits_topk_sampled_int8,
      py::arg("q"),
      py::arg("k"),
      py::arg("weights"),
      py::arg("kv_scale"),
      py::arg("starts"),
      py::arg("ends"),
      py::arg("offsets"),
      py::arg("sorted_values") = py::none(),
      py::arg("next_thresholds") = py::none(),
      py::arg("stats") = py::none(),
      py::arg("parallel_histogram") = true,
      py::arg("fallback_workspace_rows") = 0,
      py::arg("reuse_full_window") = true,
      py::arg("sort_topk") = true,
      py::arg("min_sample_k") = 131072,
      py::arg("direct_candidate_emit") = false);

  // Public MQA bindings.
  m.def("mqa_logits", &at::native::mqa_logits, "mqa_logits",
      py::arg("Q"),
      py::arg("K"),
      py::arg("Weights"),
      py::arg("KS"),
      py::arg("KE"),
      py::arg("seq_len"),
      py::arg("kv_seq_len"),
      py::arg("num_heads"),
      py::arg("head_dim"),
      py::arg("KV_scale") = torch::nullopt, 
      py::arg("clean_logits") = true,
      py::arg("D_out") = torch::nullopt);

  // Public MQA bindings.
  m.def("get_paged_mqa_logits_metadata", &lightop::deep_gemm::get_paged_mqa_logits_metadata,"get_paged_mqa_logits_metadata");
  m.def("dsv41_mqa_query_lens", &lightop::fp4_mqa::dsv41_mqa_query_lens,
        py::arg("extend_start_loc"), py::arg("extend_seq_lens"),
        py::arg("cp_size"), py::arg("cp_rank"));
  m.def("dsv41_mqa_prefill_metadata", &lightop::fp4_mqa::dsv41_mqa_prefill_metadata,
        py::arg("seq_lens"), py::arg("q_lens"), py::arg("positions"),
        py::arg("req_pool_indices"), py::arg("req_to_token"), py::arg("ratio"),
        py::arg("num_k_slots"), py::arg("max_request_work"));
  m.def("dsv41_gather_index_k_fp4", &lightop::fp4_mqa::dsv41_gather_index_k_fp4,
        py::arg("cache"), py::arg("slots"), py::arg("page_size"));
  m.def("fp8_fp4_mqa_logits", &lightop::fp4_mqa::fp8_fp4_mqa_logits,
        py::arg("q"), py::arg("kv"), py::arg("weights"),
        py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
        py::arg("clean_logits"), py::arg("max_seqlen_k"),
        py::arg("candidate_blocks") = std::nullopt,
        py::arg("candidate_query_lens") = std::nullopt);
  m.attr("_fp8_fp4_mqa_logits_supports_candidates") = true;
  m.def("paged_mqa_logits", &lightop::deep_gemm::paged_mqa_logits,"paged_mqa_logits");
  m.def("paged_mqa_logits_persisten", &lightop::deep_gemm::paged_mqa_logits_persisten,
        "dense persistent paged MQA logits with per-request KV reuse",
        py::arg("q"), py::arg("fused_kv_cache"), py::arg("weights"),
        py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
        py::arg("max_context_len"), py::arg("clean_logits") = true,
        py::arg("num_warps") = 4, py::arg("persistent_ctas") = 0,
        py::arg("group_size") = 1);
  m.def("paged_mqa_logits_fp4", &lightop::deep_gemm::paged_mqa_logits_fp4,
        "DeepGEMM fp8_fp4_paged_mqa_logits (clean_logits=False) on the fused FP4 index-K pool",
        py::arg("q"), py::arg("q_sf"), py::arg("kv_cache"), py::arg("weights"),
        py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
        py::arg("max_context_len"), py::arg("table_block_size") = 64,
        py::arg("out") = std::nullopt, py::arg("rows_per_request") = 1);
  m.attr("PAGED_MQA_LOGITS_FP4_PAIR_MIN_REQUESTS") = lightop::deep_gemm::kPagedMqaLogitsFp4PairMinRequests;
  m.attr("PAGED_MQA_LOGITS_FP4_MAX_SCHEDULE_ROWS") = lightop::deep_gemm::kPagedMqaLogitsFp4MaxScheduleRows;
  m.def("get_paged_mqa_logits_fp4_metadata",
        &lightop::deep_gemm::get_paged_mqa_logits_fp4_metadata,
        "FP4 paged MQA split-KV schedule planner",
        py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"),
        py::arg("rows_per_request") = 1);
  // --- shared sparse paged MQA group ---
  lightop::mqa_bindings::bind_paged_sparse_mqa(m);
  // --- verbatim from csrc/pybind/export.cpp (defined in paged_mqa_logits/) ---
  m.def("dsv41_candidate_block_table", &lightop::deep_gemm::dsv41_candidate_block_table,
        "DSV4.1 candidate indexer level one: top blocks of 8 by block max as pool slots / 8",
        py::arg("logits"), py::arg("seq_lens"), py::arg("page_table"), py::arg("page_size"),
        py::arg("block_size"), py::arg("out_phys_blocks"), py::arg("out_valid_lens"));
  m.def("dsv41_select_candidate_blocks", &lightop::deep_gemm::dsv41_select_candidate_blocks,
        "DSV4.1 candidate block selection and dense mask expansion",
        py::arg("logits"), py::arg("seq_lens"), py::arg("block_size"),
        py::arg("out_mask"), py::arg("out_block_ids"));
  // NOTE: export.cpp:1573-1582 (dsv41_build_swa_token_ids,
  // dsv41_remap_indices_to_staging, dsv41_ragged_topk_postprocess,
  // dsv41_pair_partners) are defined in root csrc/dsv41_*.cu /
  // dsv4_topk_workspace.cu and intentionally stay in the monolith.

  // Public MQA bindings.
  m.def(
      "dequantize_index_k_int8_paged",
      &lightop::deep_gemm::dequantize_index_k_int8_paged,
      "dequantize_index_k_int8_paged",
      py::arg("packed_cache"),
      py::arg("block_tables"),
      py::arg("context_lens"),
      py::arg("workspace"),
      py::arg("page_claims"),
      py::arg("waves_per_page") = 1);
  m.def(
      "page_mqa_logits_sparse_mask",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask,
      "page_mqa_logits_sparse_mask",
      py::arg("q"),
      py::arg("fused_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0);
  m.def(
      "page_mqa_logits_sparse_mask_grouped",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask_grouped,
      "page_mqa_logits_sparse_mask_grouped",
      py::arg("q"),
      py::arg("fused_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0,
      py::arg("group_size") = 5);
  m.def(
      "page_mqa_logits_sparse_mask_grouped_int8",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask_grouped_int8,
      "page_mqa_logits_sparse_mask_grouped_int8",
      py::arg("q"),
      py::arg("packed_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0,
      py::arg("group_size") = 5);
  // --- shared prefill sparse fusion group ---
  lightop::mqa_bindings::bind_prefill_sparse_fusion(m);
}