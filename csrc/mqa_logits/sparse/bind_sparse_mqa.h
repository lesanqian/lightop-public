// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

#include <optional>

#if !defined(LIGHTOP_PREFILL_FUSION_ONLY)
#include "paged_sparse_mqa_logits.h"
#include "paged_sparse_mqa_logits_metadata.h"
#endif

#if !defined(LIGHTOP_SPARSE_MQA_ONLY)
#include "prefill_sparse_fusion.h"
#include "prefill_sparse_sort.h"
#endif

namespace lightop::mqa_bindings {

#if !defined(LIGHTOP_PREFILL_FUSION_ONLY)
inline void bind_paged_sparse_mqa(pybind11::module_& m) {
  namespace py = pybind11;
#ifdef PREFILL_SPARSE_MQA_BUILD_ID
  m.attr("PAGED_SPARSE_MQA_BUILD_ID") = PREFILL_SPARSE_MQA_BUILD_ID;
#endif
#ifdef PREFILL_SPARSE_MQA_CONSUMER_SHA256
  m.attr("PAGED_SPARSE_MQA_CONSUMER_SHA256") = PREFILL_SPARSE_MQA_CONSUMER_SHA256;
#endif
  m.attr("PAGED_SPARSE_MQA_LOGITS_ABI") = 3;
  m.attr("PAGED_SPARSE_MQA_METADATA_FORMAT") = "deepgemm_paged_sparse_v1";
  m.def("fp8_fp4_paged_sparse_mqa_logits",
        &lightop::deep_gemm::fp8_fp4_paged_sparse_mqa_logits,
        "MXFP4/MXFP8 paged sparse logits, FP32 weights/output, Prefill/Decode dispatch",
        py::arg("q"), py::arg("q_scales"), py::arg("fused_kv_cache"),
        py::arg("weights"), py::arg("metadata"), py::arg("num_max_sparse_blocks"),
        py::arg("sparse_block_kv"), py::arg("context_lens") = std::nullopt,
        py::arg("sparse_kv_block_indices") = std::nullopt,
        py::arg("output") = std::nullopt, py::arg("num_sms") = std::nullopt,
        py::arg("is_prefill") = false);
  m.attr("sparse_fp8_fp4_paged_mqa_logits") = m.attr("fp8_fp4_paged_sparse_mqa_logits");
  m.def("get_paged_sparse_mqa_logits_metadata",
        &lightop::deep_gemm::get_paged_sparse_mqa_logits_metadata,
        "DeepGEMM get_paged_sparse_mqa_logits_metadata: merged sparse KV splits and persistent schedule",
        py::arg("context_lens"), py::arg("block_table"), py::arg("indices"),
        py::arg("page_kv"), py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"),
        py::arg("sparse_block_kv"), py::arg("num_sms") = std::nullopt);
  m.def("get_paged_sparse_mqa_logits_metadata_bytes",
        &lightop::deep_gemm::get_paged_sparse_mqa_logits_metadata_bytes,
        "Size of get_paged_sparse_mqa_logits_metadata's output",
        py::arg("num_q_tokens"), py::arg("num_max_sparse_blocks"),
        py::arg("sparse_block_kv"), py::arg("split_kv"), py::arg("num_sms"));
  m.def("prefill_sparse_fp8_fp4_paged_mqa_logits",
        &lightop::deep_gemm::prefill_sparse_fp8_fp4_paged_mqa_logits,
        py::arg("q"), py::arg("scales"), py::arg("kv"), py::arg("weights"),
        py::arg("lens"), py::arg("table"), py::arg("indices"),
        py::arg("candidates"), py::arg("capacity"), py::arg("block"),
        py::arg("num_sms") = std::nullopt, py::arg("paired_rows") = false,
        py::arg("threads") = 0);
}
#endif

#if !defined(LIGHTOP_SPARSE_MQA_ONLY)
inline void bind_prefill_sparse_fusion(pybind11::module_& m) {
#ifdef PREFILL_FUSION_BUILD_ID
  m.attr("PREFILL_SPARSE_FUSION_BUILD_ID") = PREFILL_FUSION_BUILD_ID;
#endif
  m.attr("PREFILL_SPARSE_SORT_ABI") = 1;
  m.attr("PREFILL_SPARSE_FUSION_ABI") = 1;
  m.def("prefill_sparse_native_prepare_query", &prefill_sparse_native_prepare_query);
  m.def("prefill_sparse_native_prepare_candidates", &prefill_sparse_native_prepare_candidates);
  m.def("prefill_sparse_native_finalize", &prefill_sparse_native_finalize);
  m.def("prefill_sparse_sort_candidates", &prefill_sparse_sort_candidates);
  m.def("prefill_sparse_finalize_indices", &prefill_sparse_finalize_indices);
}
#endif

}  // namespace lightop::mqa_bindings
