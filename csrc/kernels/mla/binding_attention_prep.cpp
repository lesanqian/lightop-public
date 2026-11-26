// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for attention preparation (RoPE, MLA prep, DSV4.1 SWA/pair-pool, mHC CP).

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/macros/Macros.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace lightop::dsv41 {
void fill_swa_ring_mapping(torch::Tensor mapping, const torch::Tensor& requests,
    const torch::Tensor& prefixes, const torch::Tensor& table,
    const c10::optional<torch::Tensor>& positions,
    const c10::optional<torch::Tensor>& locations,
    int64_t window, int64_t ring, int64_t block);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> build_swa_token_ids(
    const torch::Tensor& seq, const torch::Tensor& extend, const torch::Tensor& req,
    const torch::Tensor& table, const torch::Tensor& full_to_swa, int64_t window, int64_t total);
torch::Tensor remap_indices_to_staging(const torch::Tensor& indices,
    const torch::Tensor& pages, int64_t page_size, int64_t max_pages, bool static_pages);
std::tuple<torch::Tensor, torch::Tensor> pair_partners(
    const torch::Tensor& kv, const torch::Tensor& score,
    const torch::Tensor& req, const torch::Tensor& pos, const torch::Tensor& pad,
    torch::Tensor& state, int64_t ring);
}  // namespace lightop::dsv41

namespace at::native {
void spark_mla_rope_cat_store(
    const torch::Tensor& positions, const torch::Tensor& query_nope,
    const torch::Tensor& query_rope, const torch::Tensor& key_nope,
    const torch::Tensor& key_rope, const torch::Tensor& cosine_sine,
    const torch::Tensor& locations, torch::Tensor query_out,
    torch::Tensor key_rope_out, torch::Tensor cache, bool neox,
    double quant_epsilon, int64_t heads_per_block);
void indexer_k_norm_rope_hadamard_store(
    const torch::Tensor &key, const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias, const torch::Tensor &positions,
    const torch::Tensor &cos_sin, const torch::Tensor &locations,
    torch::Tensor remote_cache, torch::Tensor owner_cache, bool neox,
    double norm_epsilon, double hadamard_scale, double quant_epsilon);
std::tuple<torch::Tensor, torch::Tensor> indexer_q_rope_hadamard_quant(
    const torch::Tensor& query, const torch::Tensor& positions,
    const torch::Tensor& cos_sin, const torch::Tensor& raw_gate, bool neox,
    double hadamard_scale, double head_scale, double softmax_scale);
torch::Tensor mla_q_absorb_rope_concat(
    torch::Tensor x, torch::Tensor weight, torch::Tensor q_rope,
    torch::Tensor positions, torch::Tensor cos_sin, bool is_neox
);
Tensor RoPE_forward_gpu(int sq, int b, int np, int hn, Tensor tensor_q,Tensor tensor_cos, Tensor tensor_sin);
Tensor RoPE_backward_gpu(int sq, int b, int np, int hn,Tensor tensor_grad_out, Tensor tensor_cos, Tensor tensor_sin);
void rotary_embedding_fuse(torch::Tensor& positions,  torch::Tensor& query, std::optional<torch::Tensor> key,int64_t head_size,torch::Tensor& cos_sin_cache, bool is_neox);
void rotary_embedding_deepseek_fuse(torch::Tensor& positions,  torch::Tensor& query, std::optional<torch::Tensor> key,int64_t head_size,torch::Tensor& cos_sin_cache, bool is_neox, std::optional<torch::Tensor> offsets, bool round_bf16_products, std::optional<torch::Tensor> query_out, std::optional<torch::Tensor> key_out);
void mha_norm_rope_qkv_inplace_cuda(at::Tensor qkv, c10::optional<at::Tensor> gamma_q, c10::optional<at::Tensor> beta_q, c10::optional<at::Tensor> gamma_k,c10::optional<at::Tensor> beta_k,at::Tensor cos, at::Tensor sin, double eps);
void cross_norm_rope_q_kv_inplace_cuda(at::Tensor q,at::Tensor kv,  c10::optional<at::Tensor> gamma_q, c10::optional<at::Tensor> beta_q,c10::optional<at::Tensor> gamma_k,c10::optional<at::Tensor> beta_k, at::Tensor cos,  at::Tensor sin,  double eps);
std::vector<torch::Tensor> ds_fused_qkv_cast_fp8(
  torch::Tensor q,
  torch::Tensor kv_nope, 
  torch::Tensor k_pe,   
  int64_t qk_nope_head_dim, 
  int64_t v_head_dim
);
void dsv41_rope_fake_quant_store(
    const Tensor& input, const Tensor& freqs, Tensor& cache,
    const Tensor& indices, int64_t page_size);
Tensor dsv41_pool_pairs_norm(
    const Tensor& previous, const Tensor& current,
    const Tensor& previous_score, const Tensor& score, const Tensor& weight, double eps);
bool supports_mhc_repeat_cp_sglang(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank);
torch::Tensor mhc_repeat_cp_sglang(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank);
torch::Tensor& mhc_repeat_cp_sglang_out(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank, torch::Tensor& out);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill_swa_ring_mapping", &lightop::dsv41::fill_swa_ring_mapping,
          py::arg("mapping"), py::arg("requests"), py::arg("prefixes"),
          py::arg("table"), py::arg("positions") = py::none(),
          py::arg("locations") = py::none(), py::arg("window") = 512,
          py::arg("ring") = 1024, py::arg("block") = 8);
  m.def("spark_mla_rope_cat_store", &at::native::spark_mla_rope_cat_store,
          "Spark MLA block-specialized BF16 RoPE, concat and FP8 KV store",
          py::arg("positions"), py::arg("query_nope"), py::arg("query_rope"),
          py::arg("key_nope"), py::arg("key_rope"), py::arg("cosine_sine"),
          py::arg("locations"), py::arg("query_out"), py::arg("key_rope_out"),
          py::arg("cache"), py::arg("neox"), py::arg("quant_epsilon") = 1e-6,
          py::arg("heads_per_block") = 4);
  m.def("indexer_k_norm_rope_hadamard_store", &at::native::indexer_k_norm_rope_hadamard_store,
            "Indexer K LayerNorm, BF16 RoPE, Hadamard and mirrored INT8 cache store");
  m.def("indexer_q_rope_hadamard_quant", &at::native::indexer_q_rope_hadamard_quant,
            "Indexer Q RoPE/Hadamard/INT8 preparation with FP32 head gates");
  m.def("mla_q_absorb_rope_concat", &at::native::mla_q_absorb_rope_concat,
            "BF16 MLA Q-absorb, Q RoPE and Q concat for BS2/3/4");
  m.def("rotary_positional_embedding_forward", &at::native::RoPE_forward_gpu, "rotary positional embedding forward for gpu");
  m.def("rotary_positional_embedding_backward", &at::native::RoPE_backward_gpu, "rotary positional embedding backward for gpu");
  m.def("rotary_embedding_fuse", &at::native::rotary_embedding_fuse, "rotary_embedding_fuse",py::arg("positions"),py::arg("query"),py::arg("key"),py::arg("head_size"),py::arg("cos_sin_cache"),py::arg("is_neox"));
  m.def("rotary_embedding_deepseek_fuse",
          &at::native::rotary_embedding_deepseek_fuse,
          "RoPE: legacy in-place, or explicit outputs with rounded BF16 products",
          py::arg("positions"), py::arg("query"), py::arg("key"),
          py::arg("head_size"), py::arg("cos_sin_cache"), py::arg("is_neox"),
          py::arg("offsets") = py::none(), py::kw_only(),
          py::arg("round_bf16_products") = false,
          py::arg("query_out") = py::none(), py::arg("key_out") = py::none());
  m.def("mha_norm_rope_qkv_inplace_cuda", &at::native::mha_norm_rope_qkv_inplace_cuda,"mha_norm_rope_qkv_inplace_cuda");
  m.def("cross_norm_rope_q_kv_inplace_cuda", &at::native::cross_norm_rope_q_kv_inplace_cuda,"cross_norm_rope_q_kv_inplace_cuda");
  m.def("dsv41_build_swa_token_ids", &lightop::dsv41::build_swa_token_ids,
          "SWA metadata prefix and physical token gather");
  m.def("dsv41_remap_indices_to_staging", &lightop::dsv41::remap_indices_to_staging,
          "Inverse page map and fused staging token remap");
  m.def("dsv41_pair_partners", &lightop::dsv41::pair_partners,
          "DSV4.1 pair-partner gather and ordered ring-state update",
          py::arg("kv"), py::arg("score"), py::arg("req"), py::arg("pos"),
          py::arg("pad"), py::arg("state"), py::arg("ring_size"));
  m.def("ds_fused_qkv_cast_fp8", &at::native::ds_fused_qkv_cast_fp8, "ds_fused_qkv_cast_fp8");
  m.def("dsv41_rope_fake_quant_store", &at::native::dsv41_rope_fake_quant_store,
         "DSV41 BF16 RoPE, compressed-KV fake quantization and V4 paged store");
  m.def("dsv41_pool_pairs_norm", &at::native::dsv41_pool_pairs_norm,
         "DSV41 CP pair softmax pooling, BF16 round-trip and RMS normalization");
  m.def("supports_mhc_repeat_cp_sglang", &at::native::supports_mhc_repeat_cp_sglang,
         "Metadata-only SGLang mHC repeat+CP preflight",
         py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"), py::arg("cp_rank"));
  m.def("mhc_repeat_cp_sglang", &at::native::mhc_repeat_cp_sglang,
         "SGLang mHC CP-selected row broadcast",
         py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"), py::arg("cp_rank"));
  m.def("mhc_repeat_cp_sglang_out", &at::native::mhc_repeat_cp_sglang_out,
         "SGLang mHC CP-selected row broadcast into a non-overlapping output",
         py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"),
         py::arg("cp_rank"), py::arg("out"));
  m.attr("ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI") = 2;
}
