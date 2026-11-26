// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Sole binding owner for the KV-cache, paged dequantization, and metadata family.

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/macros/Macros.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace at::native {
void store_kv_cache(torch::Tensor packed_qkv,torch::Tensor& K_cache,torch::Tensor& V_cache,torch::Tensor q_lens,torch::Tensor accum_q_lens,
                              torch::Tensor cache_lens,torch::Tensor cache_slot_ids,torch::Tensor k_scale,torch::Tensor v_scale,const std::string& kv_cache_dtype_str,int q_head_num,int kv_head_num);
void store_kv_cache_paged(const torch::Tensor& key, const torch::Tensor& value,torch::Tensor& K_cache, torch::Tensor& V_cache,const torch::Tensor& q_lens, const torch::Tensor& accum_q_lens,
                          const torch::Tensor& cache_lens, const torch::Tensor& block_table,const torch::Tensor& k_scale, const torch::Tensor& v_scale,const std::string& kv_cache_dtype_str);
void reshape_and_cache_cuda(torch::Tensor& key,torch::Tensor& value, torch::Tensor& key_cache, torch::Tensor& value_cache, torch::Tensor& slot_mapping, const std::string& kv_cache_dtype, torch::Tensor& k_scale, torch::Tensor& v_scale);
void gather_convert_fp8_cache(
    torch::Tensor const& src_cache,    
    torch::Tensor const& dst,          
    torch::Tensor const& block_table,  
    torch::Tensor const& cu_seq_lens,  
    int64_t batch_size,
    double scale,                      
    const std::string& kv_cache_dtype, 
std::optional<torch::Tensor> seq_starts = std::nullopt);
void gather_and_maybe_dequant_cache(
    const torch::Tensor& src_cache,
    const torch::Tensor& dst,
    const torch::Tensor& block_table,
    const torch::Tensor& cu_seq_lens,
    const torch::Tensor& token_to_seq,
    int64_t num_tokens,
    const std::string& kv_cache_dtype,
    const torch::Tensor& scale,
    std::optional<torch::Tensor> seq_starts);
void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size);
void prefill_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,
    torch::Tensor const& page_table_flat,
    torch::Tensor const& kv_dst);
void decode_gather_and_up_convert(
    torch::Tensor const& src_cache,
    torch::Tensor const& token_ids,
    torch::Tensor const& dst,
    torch::Tensor const& valid_lengths);
void decode_gather_and_up_convert_with_indices(
    torch::Tensor const& src_cache,
    torch::Tensor const& token_ids,
    torch::Tensor const& dst,
    torch::Tensor const& valid_lengths,
    torch::Tensor const& compact_indices,
    int64_t tokens_per_request,
    std::optional<torch::Tensor> dedup_table,
    bool reuse_compact_indices);
void dequantize_k_cache_paged_hcu(
    const Tensor& cache, const Tensor& token_ids, Tensor& output,
    int64_t page_size, bool fp8_fnuz);
void dequantize_k_cache_paged_hcu_selected(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, torch::Tensor&, int64_t, bool);
torch::Tensor dsv4_dequantize_sparse_topk_out(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, torch::Tensor&, int64_t, int64_t, bool);
void dequantize_k_cache_paged_hcu_counted(
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&,
    int64_t, bool, const torch::Tensor&);
std::vector<torch::Tensor> dsv4_pack_c4_topk_workspace(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    int64_t, int64_t);
std::vector<torch::Tensor> dsv4_pack_sparse_topk_workspace(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    int64_t, int64_t, int64_t);
void dsv4_dequantize_k_cache_paged_out(
    torch::Tensor const& cache_u8,
    torch::Tensor const& page_table,
    torch::Tensor const& out,
    int64_t page_size,
    std::optional<torch::Tensor> valid_count);
void dsv4_gather_upconvert_k_cache_paged(
    torch::Tensor const& cache,
    torch::Tensor const& indices,
    torch::Tensor const& lengths,
    torch::Tensor const& dst,
    torch::Tensor const& compact,
    int64_t page_size);
void dsv4_gather_upconvert_dual_k_cache_paged(
    torch::Tensor const& swa_cache,
    torch::Tensor const& swa_indices,
    torch::Tensor const& swa_lengths,
    torch::Tensor const& swa_dst,
    torch::Tensor const& swa_compact,
    int64_t swa_page_size,
    torch::Tensor const& extra_cache,
    torch::Tensor const& extra_indices,
    torch::Tensor const& extra_lengths,
    torch::Tensor const& extra_dst,
    torch::Tensor const& extra_compact,
    int64_t extra_page_size);
void fused_quantize_and_store_mla_kv_cache(
    torch::Tensor k_nope,  
    torch::Tensor k_rope,    
    torch::Tensor kv_buffer, 
    torch::Tensor loc,        
    const std::string& fp8_dtype = "fp8_e4m3",
    float eps = 1e-5f
);
void fused_quantize_and_store_mla_kv_cache_dual(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, float);
torch::Tensor main_kv_loc_to_compact(torch::Tensor, torch::Tensor, int64_t);
void fused_concat_and_store_mla_kv_cache(
    torch::Tensor k_nope,
    torch::Tensor k_rope,
    torch::Tensor kv_buffer,
    torch::Tensor loc,
    const std::string& fp8_dtype = "fp8_e4m3"
);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_nope_fp8_rope_bf16_pack(
    torch::Tensor k_bf16,
    float eps
);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
    Tensor& q,
    const Tensor& kv,
    const Tensor& kv_norm_weight,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void fused_deepseek_v4_qnorm_rope_kv_rope(
    Tensor& q,
    Tensor& kv,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps);
void fused_deepseek_v4_qnorm_rope_kvnorm_rope(
    Tensor& q,
    Tensor& kv,
    const Tensor& kv_norm_weight,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& q_position_ids,
    const Tensor& kv_position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& q_position_ids,
    const Tensor& kv_position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void quantize_nope_fp8_rope_bf16_pack_store(
    torch::Tensor k_bf16,
    torch::Tensor buf,
    torch::Tensor loc,
    int page_size,
    float eps
);
void fused_dsa_cp_prefill_metadata_launcher(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const std::vector<torch::Tensor>&,
    int64_t, int64_t, int64_t, int64_t);
void fused_metadata_kernel_general_launcher(
    Tensor seq_lens,
    Tensor req_to_token,
    Tensor req_pool_indices,
    Tensor cache_seqlens_int32,
    Tensor cu_seqlens_k,
    Tensor page_table,
    const c10::optional<Tensor>& swa_page_table,
    const c10::optional<Tensor>& full_to_swa_mapping,
    int64_t B,
    int64_t max_seq_pages,
    int64_t page_size,
    int64_t seq_len_delta,
    bool    use_swa);
void lightop_hcu_alloc_extend_kernel(
    const Tensor pre_lens_ptr,
    const Tensor seq_lens_ptr,
    const Tensor last_loc_ptr,
    const Tensor free_page_ptr,
    Tensor out_indices,
    int64_t bs,
    int64_t page_size);
void minimax_cp_cache_scatter(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("store_kv_cache", &at::native::store_kv_cache, "store_kv_cache");
  m.def("store_kv_cache_paged", &at::native::store_kv_cache_paged, "store_kv_cache_paged");
  m.def("reshape_and_cache_cuda", &at::native::reshape_and_cache_cuda,"reshape_and_cache_cuda");
  m.def("gather_convert_fp8_cache", &at::native::gather_convert_fp8_cache, "gather_convert_fp8_cache");
  m.def(
       "gather_and_maybe_dequant_cache",
       &at::native::gather_and_maybe_dequant_cache,
       "Gather paged KV cache. Supported kv_cache_dtype values are auto, "
       "fp8/fp8_e4m3, fp8_ds_mla, and fp8_e5m2. FP8 inputs are "
       "dequantized as decode(cache) * scale to the FP32, FP16, or BF16 "
       "dtype selected by dst.",
       py::arg("src_cache"),
       py::arg("dst"),
       py::arg("block_table"),
       py::arg("cu_seq_lens"),
       py::arg("token_to_seq"),
       py::arg("num_tokens"),
       py::arg("kv_cache_dtype"),
       py::arg("scale"),
       py::arg("seq_starts") = py::none());
  m.def("cp_gather_and_upconvert_fp8_kv_cache", &at::native::cp_gather_and_upconvert_fp8_kv_cache, "cp_gather_and_upconvert_fp8_kv_cache");
  m.def("prefill_gather_and_upconvert_fp8_kv_cache", &at::native::prefill_gather_and_upconvert_fp8_kv_cache, "prefill_gather_and_upconvert_fp8_kv_cache");
  m.def("decode_gather_and_up_convert", &at::native::decode_gather_and_up_convert, "decode_gather_and_up_convert");
  m.def(
       "decode_gather_and_up_convert_with_indices",
       &at::native::decode_gather_and_up_convert_with_indices,
       "Decode gather/up-convert with optional request-local KV deduplication",
       py::arg("src_cache"),
       py::arg("token_ids"),
       py::arg("dst"),
       py::arg("valid_lengths"),
       py::arg("compact_indices"),
       py::arg("tokens_per_request") = 1,
       py::arg("dedup_table") = py::none(),
       py::arg("reuse_compact_indices") = false);
  m.def("dequantize_k_cache_paged_hcu", &at::native::dequantize_k_cache_paged_hcu,
         "V4 packed paged KV dequantization into preallocated BF16 output");
  m.def("dequantize_k_cache_paged_hcu_selected", &at::native::dequantize_k_cache_paged_hcu_selected,
        py::arg("cache"), py::arg("token_ids"), py::arg("selected_rows"), py::arg("valid_count"),
        py::arg("out"), py::arg("page_size"), py::arg("fp8_fnuz"));
  m.def("dsv4_dequantize_sparse_topk_out", &at::native::dsv4_dequantize_sparse_topk_out,
        py::arg("cache"), py::arg("full_ids"), py::arg("indices"), py::arg("lengths"),
        py::arg("out"), py::arg("topk"), py::arg("page_size"), py::arg("fp8_fnuz"));
  m.def("dequantize_k_cache_paged_hcu_counted",
         &at::native::dequantize_k_cache_paged_hcu_counted,
         "Dequantize the GPU-counted valid KV prefix into preallocated BF16 output",
         py::arg("cache"), py::arg("token_ids"), py::arg("out"),
         py::arg("page_size"), py::arg("fp8_fnuz"), py::arg("valid_count"));
  m.def("dsv4_pack_c4_topk_workspace",
        &at::native::dsv4_pack_c4_topk_workspace,
        "Pack a DSV4 C4 TopK union and remap query indices on GPU",
        py::arg("full_ids"), py::arg("combined_indices"),
        py::arg("combined_lens"), py::arg("topk"), py::arg("capacity"));
  m.def("dsv4_pack_sparse_topk_workspace",
        &at::native::dsv4_pack_sparse_topk_workspace,
        "Pack a C1/C2/C4 TopK union and rebase compressed/SWA indices on GPU",
        py::arg("full_ids"), py::arg("combined_indices"),
        py::arg("combined_lens"), py::arg("topk"), py::arg("capacity"),
        py::arg("swa_rows") = -1);
  m.def(
       "dsv4_dequantize_k_cache_paged_out",
       &at::native::dsv4_dequantize_k_cache_paged_out,
       "DSV4 flat E4M3FN/UE8M0 paged dequantization into preallocated BF16 output",
       py::arg("cache_u8"), py::arg("page_table"), py::arg("out"), py::arg("page_size"),
       py::arg("valid_count") = py::none());
  m.def(
       "dsv4_gather_upconvert_k_cache_paged",
       &at::native::dsv4_gather_upconvert_k_cache_paged,
       "DPSKV4 single-cache paged FP8/UE8M0 gather, BF16 upconvert, and compact indices");
  m.def(
       "dsv4_gather_upconvert_dual_k_cache_paged",
       &at::native::dsv4_gather_upconvert_dual_k_cache_paged,
       "DPSKV4 fused dual-cache paged gather, BF16 upconvert, and compact indices");
  m.def("fused_quantize_and_store_mla_kv_cache", &at::native::fused_quantize_and_store_mla_kv_cache, "fused_quantize_and_store_mla_kv_cache");
  m.def("fused_quantize_and_store_mla_kv_cache_dual", &at::native::fused_quantize_and_store_mla_kv_cache_dual, py::arg("k_nope"), py::arg("k_rope"), py::arg("kv_buffer"), py::arg("loc"), py::arg("mirror_buffer") = py::none(), py::arg("mirror_loc") = py::none(), py::arg("concat_out") = py::none(), py::arg("eps") = 1e-6f);
  m.def("main_kv_loc_to_compact", &at::native::main_kv_loc_to_compact, py::arg("loc"), py::arg("physical_to_compact"), py::arg("page_size"));
  m.def("fused_concat_and_store_mla_kv_cache", &at::native::fused_concat_and_store_mla_kv_cache, "fused_concat_and_store_mla_kv_cache");
  m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert");
  m.def("quantize_nope_fp8_rope_bf16_pack", &at::native::quantize_nope_fp8_rope_bf16_pack, "quantize_nope_fp8_rope_bf16_pack");
  m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32");
  m.def("fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32", &at::native::fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32, "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32");
  m.def("fused_deepseek_v4_qnorm_rope_kv_rope", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope, "fused_deepseek_v4_qnorm_rope_kv_rope");
  m.def("fused_deepseek_v4_qnorm_rope_kvnorm_rope", &at::native::fused_deepseek_v4_qnorm_rope_kvnorm_rope, "fused_deepseek_v4_qnorm_rope_kvnorm_rope");
  m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp");
  m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32");
  m.def("quantize_nope_fp8_rope_bf16_pack_store", &at::native::quantize_nope_fp8_rope_bf16_pack_store, "quantize_nope_fp8_rope_bf16_pack_store");
  m.def("fused_dsa_cp_prefill_metadata", &at::native::fused_dsa_cp_prefill_metadata_launcher);
  m.def("fused_metadata_kernel_general", &at::native::fused_metadata_kernel_general_launcher,
         "fused_metadata_kernel_general",
         py::arg("seq_lens"),
         py::arg("req_to_token"),
         py::arg("req_pool_indices"),
         py::arg("cache_seqlens_int32"),
         py::arg("cu_seqlens_k"),
         py::arg("page_table"),
         py::arg("swa_page_table") = torch::nullopt,
         py::arg("full_to_swa_mapping") = torch::nullopt,
         py::arg("B"),
         py::arg("max_seq_pages"),
         py::arg("page_size"),
         py::arg("seq_len_delta"),
         py::arg("use_swa"));
  m.def("hcu_alloc_extend_kernel", &at::native::lightop_hcu_alloc_extend_kernel,
         "Parallel paged KV alloc-extend kernel",
         py::arg("pre_lens_ptr"),
         py::arg("seq_lens_ptr"),
         py::arg("last_loc_ptr"),
         py::arg("free_page_ptr"),
         py::arg("out_indices"),
         py::arg("bs"),
         py::arg("page_size"));
  m.def("minimax_cp_cache_scatter", &at::native::minimax_cp_cache_scatter);
  m.attr("PREFILL_GATHER_TRITON_NAN_COMPAT_VERSION") = 1;
}
