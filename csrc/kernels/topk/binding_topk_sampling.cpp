// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for Top-K selection, index transform, and probability sampling kernels.

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/macros/Macros.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "../sampling/sampling.h"
#include "../sampling/markov_gemv_argmax.h"

namespace lightop::dsv41 {
void ragged_topk_postprocess(const torch::Tensor& scores, const torch::Tensor& selected,
    const torch::Tensor& offsets, const torch::Tensor& slots, torch::Tensor& pages,
    const c10::optional<torch::Tensor>& raw, bool mask_scores);
}  // namespace lightop::dsv41

namespace at::native {
std::tuple<Tensor,Tensor> topk_opt(const Tensor& x,int k,int dim,bool sorted);
void top_k_per_row_prefill(const torch::Tensor& logits,
        const torch::Tensor& rowStarts,
        const torch::Tensor& rowEnds, torch::Tensor& indices,
        int64_t numRows, int64_t stride0, int64_t stride1,
        int64_t topK);
void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
        const torch::Tensor& seqLens, torch::Tensor& indices,
        int64_t numRows, int64_t stride0, int64_t stride1,
        int64_t topK);
torch::Tensor convert_req_index_to_global_index(
    torch::Tensor req_id,
    torch::Tensor block_table,
    torch::Tensor token_indices,
    int BLOCK_SIZE,
    int NUM_TOPK_TOKENS,
    bool HAS_PREFILL_WORKSPACE,
    std::optional<torch::Tensor> prefill_workspace_request_ids,
    std::optional<torch::Tensor> prefill_workspace_starts);
void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt,
    bool parallel_candidates);
void fast_topk_transform_sparse_mask_interface(
    const at::Tensor& score,
    const at::Tensor& nonzero_mask,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q);
void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt);
void topk_transform_ragged_v2_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& output,
    const at::Tensor& offsets,
    std::optional<at::Tensor> row_starts_opt);
void fast_kpool_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& output,
    int64_t pool_size,
    const at::Tensor& seq_lens,
    std::optional<at::Tensor> page_table_opt,
    std::optional<at::Tensor> topk_indices_offset_opt,
    std::optional<at::Tensor> row_starts_opt,
    std::optional<at::Tensor> page_table_row_index_opt);
void combine_topk_swa_indices_hcu(
    const Tensor& topk_indices, const Tensor& query_start_loc,
    const Tensor& query_pos, const Tensor& seq_lens, const Tensor& gather_lens,
    const Tensor& compressed_base, const Tensor& swa_base,
    Tensor& out_indices, Tensor& out_lens,
    int64_t window_size, int64_t compress_ratio, int64_t topk);
void topk_ids_postprocess(torch::Tensor topk_ids, torch::optional<torch::Tensor> logical_to_physical_map, torch::optional<torch::Tensor> num_token_non_padded);
void topk_transform_512(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    std::optional<at::Tensor> out_raw_indices_opt);
void topk_transform_1024(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    std::optional<at::Tensor> out_raw_indices_opt);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_opt", &at::native::topk_opt, "topk_opt");
  m.def("dsv41_ragged_topk_postprocess", &lightop::dsv41::ragged_topk_postprocess,
          "Score masking, ascending index sort and ragged physical/raw outputs");
  m.def("markov_gemv_add_argmax", &lightop::sampling::markov_gemv_add_argmax,
          py::arg("latent"), py::arg("weight"), py::arg("base_logits"),
          "BF16 rank-256 Markov GEMV + BF16/FP32 logits add + greedy argmax (BS1-4)");
  m.def("top_k_sampling_from_probs", &lightop::sampling::top_k_sampling_from_probs, "top_k_sampling_from_probs");
  m.def("top_p_sampling_from_probs", &lightop::sampling::top_p_sampling_from_probs, "top_p_sampling_from_probs");
  m.def("top_k_top_p_sampling_from_probs", &lightop::sampling::top_k_top_p_sampling_from_probs, "top_k_top_p_sampling_from_probs");
  m.def("top_k_first_top_p_sampling_from_probs", &lightop::sampling::top_k_first_top_p_sampling_from_probs, "top_k_first_top_p_sampling_from_probs");
  m.def("top_k_per_row_prefill", &at::native::top_k_per_row_prefill, "top_k_per_row_prefill");
  m.def("top_k_per_row_decode", &at::native::top_k_per_row_decode, "top_k_per_row_decode");
  m.def("convert_req_index_to_global_index", &at::native::convert_req_index_to_global_index, "convert_req_index_to_global_index");
  m.def("fast_topk_transform_interface", &at::native::fast_topk_transform_interface,
          "transform length TopK indices into paged KVCache",
          py::arg("score"), py::arg("lengths"), py::arg("dst_page_table"),
          py::arg("src_page_table"), py::arg("cu_seqlens_q"),
          py::arg("row_starts_opt"), py::arg("parallel_candidates") = false);
  m.def(
       "fast_topk_transform_sparse_mask_interface",
       &at::native::fast_topk_transform_sparse_mask_interface,
       "fast_topk_transform_sparse_mask_interface");
  m.def("fast_topk_transform_ragged_interface", &at::native::fast_topk_transform_ragged_interface, "fast_topk_transform_ragged_interface");
  m.def("topk_transform_ragged_v2_interface", &at::native::topk_transform_ragged_v2_interface, "Fused HIP ragged TopK and additive index transform");
  m.def(
       "fast_kpool_topk_transform_interface",
       &at::native::fast_kpool_topk_transform_interface,
       "Fused kpool topk, token expansion, index transform, and tail append",
       py::arg("score"),
       py::arg("lengths"),
       py::arg("output"),
       py::arg("pool_size"),
       py::arg("seq_lens"),
       py::arg("page_table") = py::none(),
       py::arg("topk_indices_offset") = py::none(),
       py::arg("row_starts") = py::none(),
       py::arg("page_table_row_index") = py::none());
  m.def("combine_topk_swa_indices_hcu", &at::native::combine_topk_swa_indices_hcu,
         "Combine compressed TopK and positional SWA indices into a flat workspace");
  m.def("topk_ids_postprocess",&at::native::topk_ids_postprocess,"topk_ids_postprocess");
  m.def(
       "topk_transform_512",
       &at::native::topk_transform_512,
       "DeepSeek V4 fused topk-512 and page-table transform",
       py::arg("scores"),
       py::arg("seq_lens"),
       py::arg("page_tables"),
       py::arg("out_page_indices"),
       py::arg("page_size"),
       py::arg("out_raw_indices") = py::none());
  m.def(
       "topk_transform_1024",
       &at::native::topk_transform_1024,
       "DeepSeek V4 fused topk-1024 and page-table transform",
       py::arg("scores"),
       py::arg("seq_lens"),
       py::arg("page_tables"),
       py::arg("out_page_indices"),
       py::arg("page_size"),
       py::arg("out_raw_indices") = py::none());
}
