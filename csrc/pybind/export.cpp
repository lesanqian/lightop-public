// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

namespace at::native {
std::vector<torch::Tensor> minimax_qk_norm_rope(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,int64_t,bool,int64_t);
}

#if (defined(LIGHTOP_MHC_REPEAT_CP_ONLY) + defined(LIGHTOP_CP_OUTPUT_ONLY) + \
     defined(LIGHTOP_TOPK_WORKSPACE_ONLY) + defined(LIGHTOP_SPARSE_MQA_ONLY) + \
     defined(LIGHTOP_PREFILL_FUSION_ONLY)) > 1
#error "Select only one focused LightOp extension"
#endif

#if defined(LIGHTOP_MHC_REPEAT_CP_ONLY)
namespace at::native {
bool supports_mhc_repeat_cp_sglang(const torch::Tensor&, int64_t, int64_t,
                                   int64_t);
torch::Tensor mhc_repeat_cp_sglang(const torch::Tensor&, int64_t, int64_t,
                                   int64_t);
torch::Tensor& mhc_repeat_cp_sglang_out(const torch::Tensor&, int64_t, int64_t,
                                        int64_t, torch::Tensor&);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("supports_mhc_repeat_cp_sglang",
             &at::native::supports_mhc_repeat_cp_sglang,
             py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"),
             py::arg("cp_rank"));
  module.def("mhc_repeat_cp_sglang", &at::native::mhc_repeat_cp_sglang,
             py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"),
             py::arg("cp_rank"));
  module.def("mhc_repeat_cp_sglang_out", &at::native::mhc_repeat_cp_sglang_out,
             py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"),
             py::arg("cp_rank"), py::arg("out"));
}

#elif defined(LIGHTOP_CP_OUTPUT_ONLY)
namespace lightop::cp_output {
void interleave_rows_(at::Tensor, int64_t, const at::Tensor&);
void make_interleave_plan_(at::Tensor, int64_t, int64_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("interleave_rows_", &lightop::cp_output::interleave_rows_,
             py::arg("gathered"), py::arg("cp_size"), py::arg("cycle_leaders"));
  module.def("make_interleave_plan_", &lightop::cp_output::make_interleave_plan_,
             py::arg("cycle_leaders"), py::arg("rows"), py::arg("cp_size"));
}

#elif defined(LIGHTOP_TOPK_WORKSPACE_ONLY)
namespace at::native {
void dequantize_k_cache_paged_hcu(const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int64_t, bool);
void dequantize_k_cache_paged_hcu_selected(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, torch::Tensor&, int64_t, bool);
torch::Tensor dsv4_dequantize_sparse_topk_out(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, torch::Tensor&, int64_t, int64_t, bool);
void dequantize_k_cache_paged_hcu_counted(
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&,
    int64_t, bool, const torch::Tensor&);
std::vector<torch::Tensor> dsv4_pack_sparse_topk_workspace(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    int64_t, int64_t, int64_t);
std::vector<torch::Tensor> dsv4_pack_c4_topk_workspace(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    int64_t, int64_t);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dequantize_k_cache_paged_hcu", &at::native::dequantize_k_cache_paged_hcu);
  module.def("dequantize_k_cache_paged_hcu_selected", &at::native::dequantize_k_cache_paged_hcu_selected,
      py::arg("cache"), py::arg("token_ids"), py::arg("selected_rows"), py::arg("valid_count"),
      py::arg("out"), py::arg("page_size"), py::arg("fp8_fnuz"));
  module.def("dsv4_dequantize_sparse_topk_out", &at::native::dsv4_dequantize_sparse_topk_out,
      py::arg("cache"), py::arg("full_ids"), py::arg("indices"), py::arg("lengths"),
      py::arg("out"), py::arg("topk"), py::arg("page_size"), py::arg("fp8_fnuz"));
  module.def("dequantize_k_cache_paged_hcu_counted",
             &at::native::dequantize_k_cache_paged_hcu_counted,
             py::arg("cache"), py::arg("token_ids"), py::arg("out"),
             py::arg("page_size"), py::arg("fp8_fnuz"), py::arg("valid_count"));
  module.def("dsv4_pack_sparse_topk_workspace",
             &at::native::dsv4_pack_sparse_topk_workspace,
             py::arg("full_ids"), py::arg("combined_indices"),
             py::arg("combined_lens"), py::arg("topk"),
             py::arg("capacity"), py::arg("swa_rows") = -1);
  module.def("dsv4_pack_c4_topk_workspace",
             &at::native::dsv4_pack_c4_topk_workspace,
             py::arg("full_ids"), py::arg("combined_indices"),
             py::arg("combined_lens"), py::arg("topk"), py::arg("capacity"));
}

#else
#if defined(LIGHTOP_SPARSE_MQA_ONLY) && defined(LIGHTOP_PREFILL_FUSION_ONLY)
#error "Select only one focused LightOp extension"
#endif

// Local correctness/performance builds use the same shared sparse MQA bindings.
#if defined(LIGHTOP_SPARSE_MQA_ONLY) || defined(LIGHTOP_PREFILL_FUSION_ONLY)
#include "../mqa_logits/sparse/bind_sparse_mqa.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(LIGHTOP_SPARSE_MQA_ONLY)
  lightop::mqa_bindings::bind_paged_sparse_mqa(m);
#else
  lightop::mqa_bindings::bind_prefill_sparse_fusion(m);
#endif
}
#else
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <vector>
#include <optional>

namespace lightop::cp_output {
void interleave_rows_(at::Tensor, int64_t, const at::Tensor&);
void make_interleave_plan_(at::Tensor, int64_t, int64_t);
}

namespace at {  namespace native {
// Spark MLA Q/K RoPE, Q concatenation and FP8 cache write in one kernel.
// Q-only DSA Indexer preparation; the K/cache path remains independent.

// Native MLA Q-absorb with BF16 Q RoPE and direct Q concatenation.

std::optional<Tensor> hc_mix_stats(const Tensor& x, const Tensor& w, double eps);
std::tuple<Tensor, Tensor, Tensor> hc_mix_stats_sinkhorn_bf16x3(const Tensor&, const std::vector<Tensor>&, const Tensor&, const Tensor&, int64_t, double, double, int64_t);
std::tuple<Tensor, Tensor, Tensor> hc_mix_stats_sinkhorn_deepgemm(const Tensor&, const std::vector<Tensor>&, const Tensor&, const Tensor&, int64_t, double, double, int64_t);

Tensor rmsnorm_forward_autograd(const Tensor& x,const Tensor& weight,double eps,bool training);
std::tuple<Tensor,Tensor>rmsnorm_forward(const Tensor& self,const Tensor& weight,c10::optional<Tensor>& output_inplace,double eps,bool training);
std::tuple<Tensor,Tensor>rmsnorm_backward(const Tensor& grad, const Tensor& X,const Tensor& rstd,const Tensor& gamma);
void fused_rms_norm_contiguous(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);
void fused_rms_norm_rope_contiguous(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit, bool flashmla_fp8_layout, float kv_quant_epsilon, const c10::optional<torch::Tensor>& kv_smooth_scale);
void fused_rms_norm_rope_contiguous_q_cat(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed, torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, torch::Tensor& q_nope, torch::Tensor& q_cat, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit, bool flashmla_fp8_layout, float kv_quant_epsilon, const c10::optional<torch::Tensor>& kv_smooth_scale);
void fuse_rmsnorm_rope_quant_qkv(torch::Tensor& positions, torch::Tensor& q_c, torch::Tensor& q_pe, torch::Tensor& q_quant, torch::Tensor& q_scale, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit);
Tensor grid_sample_forward_autograd(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
Tensor grid_sample_forward(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
std::tuple<Tensor,Tensor> grid_sample_backward(const Tensor& grad_output, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, std::array<bool, 2> output_mask);
void fuse_fill_rms_x2_concat(torch::Tensor& hidden_states_fuse, torch::Tensor& positions, torch::Tensor& inputs_embeds, torch::Tensor& previous_hidden_states, torch::Tensor& weight_inputs_embeds, torch::Tensor& weight_previous_hidden_states, float epsilon);
//void rms_rotary_embedding_fuse_with_kv_store(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, torch::Tensor& v, torch::Tensor& cos_sin_cache, int head_dim, int page_size, torch::Tensor& k_buffer, torch::Tensor& v_buffer, Tensor& kv_cache_loc, bool interleave, Tensor& weight_q, Tensor& weight_k, torch::Dtype output_dtype, std::optional<Tensor>& residual_q, std::optional<Tensor>& residual_k, std::optional<float>& k_scale, std::optional<float>& v_scale, std::optional<Tensor>& q_rope_out, std::optional<Tensor>& k_rope_out, std::optional<Tensor>& v_rope_out,  float epsilon);
//void rms_rotary_embedding_fuse_with_kv_store(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, torch::Tensor& v, int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox, torch::Tensor& weight_q, torch::Tensor& weight_k, std::optional<Tensor>& residual_q, std::optional<Tensor>& residual_k, torch::Tensor& k_buffer, torch::Tensor& v_buffer, torch::Tensor& kv_cache_loc, torch::Tensor& q_rope_out, torch::Tensor& k_rope_out, torch::Tensor& v_rope_out, int page_size, double epsilon, bool use_fp8_e4m3);
void rms_rotary_embedding_fuse_with_kv_store(
    Tensor& positions, 
    Tensor& query, 
    Tensor& key,
    //////////////
    Tensor& v,
    //////////////
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    std::optional<Tensor>& weight_q, 
    std::optional<Tensor>& weight_k, 
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    ///////////////////////////////////////////////////
    Tensor& k_buffer,
    Tensor& v_buffer,
    Tensor& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    /////////////////////////////////////////////////
    double epsilon,
    bool use_fp8_e4m3);
void gammarms_rotary_embedding_fuse_with_kv_store(
    Tensor& positions, 
    Tensor& query, 
    Tensor& key,
    //////////////
    Tensor& v,
    //////////////
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    std::optional<Tensor>& weight_q, 
    std::optional<Tensor>& weight_k, 
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    ///////////////////////////////////////////////////
    Tensor& k_buffer,
    Tensor& v_buffer,
    Tensor& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    /////////////////////////////////////////////////
    double epsilon,
    bool use_fp8_e4m3,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    bool is_interleaved);
void split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant(
    Tensor& positions,
    Tensor& qkv,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    std::optional<Tensor>& weight_q,
    std::optional<Tensor>& weight_k,
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    Tensor& k_buffer,
    Tensor& v_buffer,
    std::optional<Tensor>& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    double epsilon,
    bool use_fp8_e4m3,
    bool use_fp8_cache);
Tensor fuse_rn_dropout_forward_autograd(const Tensor& x,const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_dropout_bias_forward_autograd(const Tensor& x,const Tensor& y,const Tensor &bias, const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,bool training,bool inplace);
Tensor drop_path_forward_autograd(const Tensor& x,double p, bool training,bool scale_by_keep) ;
Tensor fuse_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,double rate,bool training);
Tensor fused_linear_bias_forward_autograd(const Tensor& input, const Tensor& weight, const Tensor& bias);
std::tuple <Tensor, Tensor> fused_relu_dropout_forward(const Tensor &self, const double &p, const bool &use_relu, const bool &training);
Tensor fused_relu_dropout_backward(Tensor &grad, Tensor &mask);
Tensor fuse_softmax_forward_autograd(const Tensor &input, const int64_t dim, const bool half_to_float, const bool training);
Tensor fuse_logsoftmax_forward_autograd(const at::Tensor &input, const int64_t dim, const bool half_to_float, const bool training);
Tensor accelerate_rnnt_loss(Tensor &logits, const Tensor &targets, const Tensor &logit_lengths, const Tensor &target_lengths, int blank, double clamp, bool compute_sum, bool compute_mean, bool check_lengths);
Tensor accelerate_ctc_loss(Tensor &logits, const Tensor &targets, const Tensor &logit_lengths, const Tensor &target_lengths, int blank, double clamp, bool logits_time_major, bool compute_sum, bool compute_mean, bool check_lengths);
Tensor fused_bias_gelu_forward(const Tensor &x, const Tensor &bias);
std::tuple <Tensor, Tensor> fused_bias_gelu_backward(const Tensor  &x, const Tensor &bias,const Tensor &grad);
Tensor fuse_scaled_upper_triang_masked_softmax_forward_autograd(torch::Tensor const& input, float scale_factor);
Tensor fuse_scaled_masked_softmax_forward_autograd(torch::Tensor const& input, torch::Tensor const& mask, float scale_factor);
int get_batch_per_block(int query_seq_len, int key_seq_len, int batches, int attn_heads);
Tensor fuse_scaled_upper_triang_masked_softmax_dropout_forward_autograd(torch::Tensor const& input, float scale_factor, float p);
Tensor fuse_scaled_masked_softmax_dropout_forward_autograd(torch::Tensor const& input, torch::Tensor const& mask, float scale_factor, double p);
Tensor FusedBiasSwiGLU_forward_gpu(Tensor in, Tensor bias);
std::vector<Tensor> FusedBiasSwiGLU_backward_gpu(Tensor out_grad,Tensor in,Tensor bias);
std::tuple <Tensor, Tensor> fuse_bias_dropout_add_forward(const Tensor &x,const Tensor &bias,const Tensor &residual,double droprate,bool training);
std::tuple <Tensor, Tensor> fuse_bias_dropout_add_backward(const Tensor &grad,const Tensor &mask, double droprate);
Tensor fused_bmm(torch::Tensor x, torch::Tensor w, torch::Tensor output, torch::Tensor x_scale, torch::Tensor w_scale, std::optional<torch::Tensor> bias, torch::Dtype dtype, bool transpose_bm, bool transpose_bm_in, int mode);
Tensor sum_opt(const Tensor& x,int dim);
Tensor mean_opt(const Tensor& x,int dim);
Tensor gemm_opt(const Tensor& x,const Tensor& y, c10::optional<at::ScalarType> out_dtype);
Tensor gemm_w4_a16(const Tensor& x,const Tensor& y,const Tensor& scale_m,const Tensor& zero_m);
Tensor w4_to_w16(const Tensor& x,const Tensor& scale_m,const Tensor& zero_m);
Tensor gemm_w8_a16_fp8_colmajor(const Tensor& x,const Tensor& y,const Tensor& scale,int warps,int n_per_block,int splitk);
void fp8to16(Tensor &ret, const Tensor &x,const Tensor& scale);
Tensor gemm_w8_a8_smooth_colmajor(const Tensor& x,const Tensor& y,const Tensor& scale_a,const Tensor& scale_b,  const std::optional<Tensor> bias, at::ScalarType out_dtype,int warps,int n_per_block, int n_per_warp, int splitk, bool use_n_warps);

Tensor ds_concat(torch::Tensor input_A, torch::Tensor input_B, torch::Tensor output, int32_t mode);

void rms_norm_dynamic_per_token_quant(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input,double limit);
void rms_norm_dynamic_per_token_quant_sglang(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input,double limit,std::optional<Tensor> smooth_scale,std::optional<Tensor> norm_output,bool clamp_norm_output,std::optional<Tensor> residual_output,std::optional<Tensor> clamped_output);
void mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(Tensor& q_out,Tensor& packed_input,const Tensor& q_weight,const Tensor& kv_weight,
                                                       Tensor q_scales,double q_epsilon,double kv_epsilon);
void rms_norm_fp8_quant(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input, int fp8type, std::optional<Tensor> norm_output);
void create_gemmbias_handle();
void destroy_gemmbias_handle();
void create_gemmdbias_handle();
void destroy_gemmdbias_handle();

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,torch::optional<torch::Tensor>& token_expert_indices,torch::Tensor& gating_output,  bool Is_renormalize = false);

//------------------------------------------------- groupgemm ---------------------------------------//
//------------------------------------------------- groupgemm ---------------------------------------//

void fuse_silu_mul_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids) ;
void fuse_silu_mul_clamp_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, double limit, bool use_gelu, const c10::optional<torch::Tensor>& smooth_scale, const c10::optional<torch::Tensor>& num_local_tokens_tensor = c10::nullopt, int expect_m = -1, bool round_silu = true);
void fuse_silu_mul_clamp_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, double limit, const torch::Tensor& mask_m, int expect_m, bool use_gelu, const c10::optional<torch::Tensor>& smooth_scale, bool round_silu);
void fuse_silu_mul_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, torch::optional<torch::Tensor> &tokens_per_expert, torch::optional<torch::Tensor> &num_local_tokens_tensor,int topK,int expert_m);
void fuse_situ_mul_quant_contiguous(torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales, double situ_beta, double situ_linear_beta);
void fuse_situ_mul_quant_ep(torch::Tensor& input, torch::Tensor& out, torch::Tensor& scales, torch::Tensor& masked_m, double situ_beta, double situ_linear_beta, int64_t expect_m);
void fuse_silu_mul_fp8_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids, float limit = -1.0f, bool per_token_scale = false);
void swiglu_oai_fp8_quant_masked(const torch::Tensor&, torch::Tensor&, torch::Tensor&, const torch::Tensor&, double, double, int64_t, int64_t);
void swiglu_oai_fp8_quant_masked_flat(const torch::Tensor&, torch::Tensor&, torch::Tensor&, const torch::Tensor&, double, double, int64_t, int64_t);
void fuse_silu_mul_fp8_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &tokens_per_expert, torch::optional<torch::Tensor> &num_local_tokens_tensor,int topK,int expert_m, float limit);
void fuse_silu_and_mul(torch::Tensor& input, torch::Tensor& out);
void fuse_silu_and_mul_ep(torch::Tensor& input, torch::Tensor& out, const torch::Tensor& mask_m, int expect_m);
Tensor ms_deform_attn_cuda_forward(const Tensor &value, const Tensor &spatial_shapes, const Tensor &level_start_index, const Tensor &sampling_loc, const Tensor &attn_weight, const int im2col_step);
void ms_deform_attn_cuda_backward(const Tensor &value, const Tensor &spatial_shapes, const Tensor &level_start_index, const Tensor &sampling_loc, const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value, Tensor &grad_sampling_loc, Tensor &grad_attn_weight, const int im2col_step);
torch::Tensor box_cost(const torch::Tensor &pred, const torch::Tensor &target, const torch::Tensor &inst_weights, const torch::Tensor &dim_weights, const float cost_weight); 
torch::Tensor encode_reg_t(const torch::Tensor &input, const torch::Tensor &lengths);
std::vector<torch::Tensor> batch_linear_assignment(torch::Tensor cost);
void fill_zeros(Tensor &out, int bs, int hidden_size);
void rms_rotary_embedding_fuse(Tensor &positions,Tensor &query,Tensor &key,int64_t head_size,Tensor &cos_sin_cache, bool is_neox,Tensor weight_q,Tensor weight_k,c10::optional<Tensor> residual_q,c10::optional<Tensor> residual_k,double epsilon);

void fuse_rms_mrope_cuda(Tensor &q, Tensor &k, const Tensor &cos, const Tensor &sin, const std::vector<int>& mrope_section, int head_size, int rotary_dim,bool is_interleaved,Tensor weight_q,Tensor weight_k,c10::optional<Tensor> residual_q,c10::optional<Tensor> residual_k,double epsilon);
torch::Tensor backward_polyfill_cuda(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,torch::Tensor grad_output,torch::Tensor grad_vertices,int width,int height,float inv_smoothness,int mode);
std::vector<torch::Tensor> forward_polyfill_cuda(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,int width,int height,float inv_smoothness,int mode);
torch::Tensor backward_polyfill_lightop(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,torch::Tensor grad_output,torch::Tensor grad_vertices,int width,int height,float inv_smoothness,int mode);
std::vector<torch::Tensor> forward_polyfill_lightop(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,int width,int height,float inv_smoothness,int mode);
void rms_norm_opt(torch::Tensor& out,torch::Tensor& input,torch::Tensor& weight,double epsilon,double limit);
void rms_norm_no_weight(
    torch::optional<torch::Tensor> out,
    torch::Tensor& input,
    torch::optional<torch::Tensor> weight,
    double epsilon);
void fused_add_rms_norm_opt(torch::Tensor& input,torch::Tensor& residual, torch::Tensor& weight,double epsilon,double limit,bool use_aligned_loads = false);
void silu_and_mul_opt(torch::Tensor& out,torch::Tensor& input);
void gelu_and_mul_opt(torch::Tensor& out,torch::Tensor& input);
void gelu_tanh_and_mul_opt(torch::Tensor& out,torch::Tensor& input); 
void relu2_opt(torch::Tensor& out,torch::Tensor& input); 
torch::Tensor reshape_to_float(torch::Tensor input, int64_t vocab_size);
torch::Tensor add_int8(const torch::Tensor& a, const torch::Tensor& b, bool saturate=true);
torch::Tensor sub_int8(const torch::Tensor& a, const torch::Tensor& b, bool saturate=true);
std::tuple<torch::Tensor, torch::Tensor> pack_padded_sequence_custom(
    const torch::Tensor& input,
    const torch::Tensor& lengths,
    bool batch_first);
//------------------------------------------------- bilinear_upsample ---------------------------------------//
torch::Tensor bilinear_upsample(const torch::Tensor& x, const std::vector<int64_t>& output_size);
torch::Tensor bilinear_upsample_backward(const torch::Tensor& grad_output, const std::vector<int64_t>& input_size);
//------------------------------------------------- bilinear_upsample ---------------------------------------//
void fused_rms_norm_rope_int8quant_contiguous(
    torch::Tensor& positions, 
    torch::Tensor& q_pe, 
    torch::Tensor& k_pe, 
    torch::Tensor& kv_c, 
    torch::Tensor& kv_c_normed,      
    torch::Tensor& kv_c_normed_int8, 
    torch::Tensor& kv_c_normed_scale, 
    torch::Tensor& weight, 
    torch::Tensor& cos_sin_cache, 
    torch::Tensor& slot_mapping, 
    torch::Tensor& kv_cache, 
    std::string& kv_cache_dtype_str, 
    float kv_cache_scale, 
    bool is_neox_style, 
    float epsilon);
void rms_norm_smooth_per_token_dynamic_quant(
  Tensor &out,            
  Tensor& input,                
  std::optional<Tensor> weight,  
  const Tensor& smooth_scale,    
  Tensor scales,                
  double epsilon,
  std::optional<Tensor> residual,
  std::optional<Tensor> after_norm,    
  std::optional<Tensor> after_residual,
  bool update_input,               
  bool update_res                   
);
void swiglu_oai_build_gate_table(torch::Tensor&,double,double);
void swiglu_oai_build_int8_quant_table(torch::Tensor&);
void swiglu_oai_int8_quant_contiguous_tables(const torch::Tensor&,torch::Tensor&,torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double,double);
void swiglu_oai_int8_quant_contiguous_block_tables(const torch::Tensor&,torch::Tensor&,torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double,double,int64_t,int64_t);
void swiglu_oai_int8_quant_contiguous_direct(const torch::Tensor&,torch::Tensor&,torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double,double,int64_t,int64_t);
void swiglu_oai_int8_quant_masked_flat_tables(const torch::Tensor&,torch::Tensor&,torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double,double,int64_t,int64_t);
void swiglu_oai_int8_quant_contiguous(const torch::Tensor&, torch::Tensor&, torch::Tensor&, double, double, double);
void swiglu_oai_int8_quant_masked(const torch::Tensor&, torch::Tensor&, torch::Tensor&, const torch::Tensor&, double, double, double);
std::vector<torch::Tensor> swiglu_oai_fp8_quant_contiguous(const torch::Tensor& input, double alpha, double beta, double limit);
void fuse_qkv_head_rms_norm(
    torch::Tensor& token_data,   
    torch::Tensor& q_weight,     
    torch::Tensor& k_weight,    
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    double epsilon); 
void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon);
void gemma_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);
void gemma_rms_norm_fp8_quant(
  Tensor &out, Tensor& input, const Tensor& weight, Tensor scales, 
  double epsilon, std::optional<Tensor> residual, bool update_input, int fp8type
);  
std::vector<Tensor> engram_hash_ids_hcu(
    const Tensor&, const Tensor&, const Tensor&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int64_t, int64_t, int64_t, int64_t, int64_t);

std::tuple<Tensor, Tensor, Tensor> qkv_split_rms_rotary_embedding_fuse(
    Tensor& positions, 
    Tensor& qkv, 
    int64_t q_size,
    int64_t kv_size,
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    Tensor weight_q, 
    Tensor weight_k, 
    std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k,
    double epsilon); 
std::tuple<Tensor, Tensor, Tensor> gemma_qkv_split_rms_rotary_embedding_fuse(
    Tensor& positions, Tensor& qkv, int64_t q_size, int64_t kv_size,
    int64_t head_size, Tensor& cos_sin_cache, bool is_neox, Tensor weight_q,
    Tensor weight_k, std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k, double epsilon); 
torch::Tensor dpskv4_hc_head_fused(torch::Tensor mixes, torch::Tensor x, torch::Tensor hc_scale, torch::Tensor hc_base, double hc_eps);
void mimo_v2_split_rope_vscale_kv_store(
    Tensor& positions,
    Tensor& qkv,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t head_size,
    int64_t v_head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    Tensor& k_buffer,
    Tensor& v_buffer,
    std::optional<Tensor>& kv_cache_loc,
    Tensor& q_out,
    Tensor& k_out,
    Tensor& v_out,
    double value_scale,
    std::optional<Tensor>& k_cache_scale,
    std::optional<Tensor>& v_cache_scale,
    bool use_fp8_e4m3,
    bool use_fp8_cache);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("grid_sample_forward_autograd", &at::native::grid_sample_forward_autograd, "grid_sample forward autograd");
  m.def("grid_sample_forward", &at::native::grid_sample_forward, "grid_sample forward");
  m.def("grid_sample_backward", &at::native::grid_sample_backward, "grid_sample backward");

  m.def("droppath_forward_autograd",&at::native::drop_path_forward_autograd,"droppath forward") ;
  m.def("add_dropout_forward_autograd", &at::native::fuse_add_dropout_forward_autograd, "fused_dropout forward");
  m.def("fused_linear_bias_forward_autograd", &at::native::fused_linear_bias_forward_autograd, "fused_linearbias forward autograd");
  m.def("relu_dropout_forward", &at::native::fused_relu_dropout_forward, "fused_relu_dropout forward");
  m.def("relu_dropout_backward", &at::native::fused_relu_dropout_backward, "fused_relu_dropout backward");
  m.def("softmax_forward_autograd", &at::native::fuse_softmax_forward_autograd, "softmax autograd of forward");
  m.def("logsoftmax_forward_autograd", &at::native::fuse_logsoftmax_forward_autograd, "logsoftmax autograd of forward");
  m.def("accelerate_rnnt_loss", &at::native::accelerate_rnnt_loss, "accelerated rnntloss");
  m.def("accelerate_ctc_loss", &at::native::accelerate_ctc_loss, "accelerated ctcloss");
  m.def("create_gemmbias_handle", &at::native::create_gemmbias_handle, "create_gemmbias_handle");
  m.def("destroy_gemmbias_handle", &at::native::destroy_gemmbias_handle, "destroy_gemmbias_handle");
  m.def("create_gemmdbias_handle", &at::native::create_gemmdbias_handle, "create_gemmdbias_handle");
  m.def("destroy_gemmdbias_handle", &at::native::destroy_gemmdbias_handle, "destroy_gemmdbias_handle");
  m.def("scaled_upper_triang_masked_softmax_forward_autograd", &at::native::fuse_scaled_upper_triang_masked_softmax_forward_autograd, "scaled_upper_masked_softmax forward");
  m.def("scaled_masked_softmax_forward_autograd", &at::native::fuse_scaled_masked_softmax_forward_autograd, "masked_softmax_forward  forward");
  m.def("scaled_upper_triang_masked_softmax_dropout_forward_autograd", &at::native::fuse_scaled_upper_triang_masked_softmax_dropout_forward_autograd, "scaled_upper_triang_masked_softmax_dropout forward");
  m.def("scaled_masked_softmax_dropout_forward_autograd", &at::native::fuse_scaled_masked_softmax_dropout_forward_autograd, "scaled_masked_softmax_dropout forward");
  m.def("get_batch_per_block",&at::native::get_batch_per_block, "get_batch_per_block");
  m.def("bias_gelu_forward", &at::native::fused_bias_gelu_forward, "fused_bias_gelu forward");
  m.def("bias_gelu_backward", &at::native::fused_bias_gelu_backward, "fused_bias_gelu backward");
  m.def("FusedBiasSwiGLU_forward", &at::native::FusedBiasSwiGLU_forward_gpu, "fused bias SwiGLU forward for gpu");
  m.def("FusedBiasSwiGLU_backward", &at::native::FusedBiasSwiGLU_backward_gpu, "fused bias SwiGLU backward for gpu");
  m.def("fused_bmm", &at::native::fused_bmm, "fused transepose bmm transpose");
  m.def("bias_dropout_add_forward", &at::native::fuse_bias_dropout_add_forward, "fuse_bias_dropout_add forward");
  m.def("bias_dropout_add_backward", &at::native::fuse_bias_dropout_add_backward, "fuse_bias_dropout_add backward");
  m.def("sum_opt", &at::native::sum_opt, "sum_opt");
  m.def("mean_opt", &at::native::mean_opt, "mean_opt");
  m.def("gemm_opt", &at::native::gemm_opt, "gemm_opt",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("out_dtype") = pybind11::none());
  m.def("gemm_w4_a16", &at::native::gemm_w4_a16, "gemm_w4_a16");
  m.def("w4_to_w16", &at::native::w4_to_w16, "w4_to_w16");
  m.def("gemm_w8_a16_fp8_colmajor", &at::native::gemm_w8_a16_fp8_colmajor, "gemm_w8_a16_fp8_colmajor");
  m.def("fp8to16", &at::native::fp8to16, "fp8to16");
  m.def("gemm_w8_a8_smooth_colmajor", &at::native::gemm_w8_a8_smooth_colmajor, "gemm_w8_a8_smooth_colmajor");
  m.def("bilinear_upsample", &at::native::bilinear_upsample, "bilinear_upsample");
  m.def("bilinear_upsample_backward", &at::native::bilinear_upsample_backward, "bilinear_upsample_backward");
  m.def("topk_softmax", &at::native::topk_softmax, "topk_softmax");

  m.def("fuse_silu_mul_quant",&at::native::fuse_silu_mul_quant,"fuse_silu_mul_quant");
  m.attr("SILU_CLAMP_QUANT_ROUNDING_VERSION") = 2;
  m.def(
      "fuse_silu_mul_clamp_quant",
      &at::native::fuse_silu_mul_clamp_quant,
      "fuse_silu_mul_clamp_quant",
      py::arg("input"),
      py::arg("output"),
      py::arg("scales"),
      py::arg("limit"),
      py::arg("use_gelu") = false,
      py::arg("smooth_scale") = py::none(),
      py::arg("num_local_tokens_tensor") = py::none(),
      py::arg("expect_m") = -1,
      py::arg("round_silu") = true);
  m.def(
      "fuse_silu_mul_clamp_quant_ep",
      &at::native::fuse_silu_mul_clamp_quant_ep,
      "fuse_silu_mul_clamp_quant_ep",
      py::arg("input"),
      py::arg("output"),
      py::arg("scales"),
      py::arg("limit"),
      py::arg("mask_m"),
      py::arg("expect_m") = -1,
      py::arg("use_gelu") = false,
      py::arg("smooth_scale") = py::none(),
      py::arg("round_silu") = true);
  m.def("fuse_silu_mul_quant_ep",&at::native::fuse_silu_mul_quant_ep,"fuse_silu_mul_quant_ep");
  m.def("fuse_situ_mul_quant_contiguous",&at::native::fuse_situ_mul_quant_contiguous,"fuse_situ_mul_quant_contiguous (kimi k3 softcap-glu + per-token int8 quant, contiguous)");
  m.def("fuse_situ_mul_quant_ep",(void (*)(torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, double, double, int64_t))&at::native::fuse_situ_mul_quant_ep,"fuse_situ_mul_quant_ep (kimi k3 softcap-glu + per-token int8 quant, ep-masked; expect_m grid clamp)",py::arg("input"), py::arg("output"), py::arg("scales"), py::arg("masked_m"),py::arg("situ_beta"), py::arg("situ_linear_beta"),py::arg("expect_m") = -1);
  m.def(
      "fuse_silu_mul_fp8_quant",
      &at::native::fuse_silu_mul_fp8_quant,
      "fuse_silu_mul_fp8_quant",
      py::arg("input"),
      py::arg("output"),
      py::arg("scales"),
      py::arg("fp8type"),
      py::arg("num_local_tokens_tensor"),
      py::arg("topK"),
      py::arg("expert_m"),
      py::arg("expert_ids"),
      py::arg("limit") = -1.0f,
      py::arg("per_token_scale") = false);
  m.def("swiglu_oai_fp8_quant_masked", &at::native::swiglu_oai_fp8_quant_masked, "Masked OAI SwiGLU with FP32 row-FP8 quantization");
  m.def("swiglu_oai_fp8_quant_masked_flat", &at::native::swiglu_oai_fp8_quant_masked_flat, "OAI row-FP8 quantization using live rows of 16 experts");
  m.def("fuse_silu_mul_fp8_quant_ep",&at::native::fuse_silu_mul_fp8_quant_ep,"fuse_silu_mul_fp8_quant_ep");
  m.def("fuse_silu_and_mul",&at::native::fuse_silu_and_mul,"fuse_silu_and_mul");
  m.def("ds_concat",&at::native::ds_concat,"ds_concat");
  m.def("ms_deform_attn_forward", &at::native::ms_deform_attn_cuda_forward,"ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &at::native::ms_deform_attn_cuda_backward,"ms_deform_attn_backward");
  m.def("box_cost", &at::native::box_cost, "box_cost");
  m.def("encode_reg_t", &at::native::encode_reg_t, "encode_reg_t");

  m.def("batch_linear_assignment", &at::native::batch_linear_assignment, "batch_linear_assignment");
  m.def("fill_zeros", &at::native::fill_zeros,"fill_zeros");

  m.def("backward_polyfill_cuda", &at::native::backward_polyfill_cuda, "backward_polyfill_cuda");
  m.def("forward_polyfill_cuda", &at::native::forward_polyfill_cuda, "forward_polyfill_cuda");
  m.def("forward_polyfill_lightop", &at::native::forward_polyfill_lightop, "forward_polyfill_lightop");
  m.def("backward_polyfill_lightop", &at::native::backward_polyfill_lightop, "backward_polyfill_lightop");

  m.def("silu_and_mul_opt", &at::native::silu_and_mul_opt, "silu_and_mul_opt");
  m.def("gelu_and_mul_opt", &at::native::gelu_and_mul_opt, "gelu_and_mul_opt");
  m.def("gelu_tanh_and_mul_opt", &at::native::gelu_tanh_and_mul_opt, "gelu_tanh_and_mul_opt");
  m.def("relu2_opt", &at::native::relu2_opt, "relu2_opt");
  m.def("reshape_to_float", &at::native::reshape_to_float, "reshape_to_float");
  m.def("add_int8", &at::native::add_int8, "add_int8", py::arg("a"), py::arg("b"), py::arg("saturate") = true);
  m.def("sub_int8", &at::native::sub_int8, "sub_int8", py::arg("a"), py::arg("b"), py::arg("saturate") = true);
  m.def("pack_padded_sequence_custom", &at::native::pack_padded_sequence_custom, "pack_padded_sequence_custom");

 m.def("swiglu_oai_build_gate_table", &at::native::swiglu_oai_build_gate_table);
 m.def("swiglu_oai_build_int8_quant_table", &at::native::swiglu_oai_build_int8_quant_table);
 m.def("swiglu_oai_int8_quant_contiguous_tables", &at::native::swiglu_oai_int8_quant_contiguous_tables);
 m.def("swiglu_oai_int8_quant_contiguous_block_tables", &at::native::swiglu_oai_int8_quant_contiguous_block_tables);
 m.def("swiglu_oai_int8_quant_contiguous_direct", &at::native::swiglu_oai_int8_quant_contiguous_direct);
 m.def("swiglu_oai_int8_quant_masked_flat_tables", &at::native::swiglu_oai_int8_quant_masked_flat_tables);
 m.def("swiglu_oai_int8_quant_contiguous", &at::native::swiglu_oai_int8_quant_contiguous, "BF16-rounded OAI row INT8 quantization");
 m.def("swiglu_oai_int8_quant_masked", &at::native::swiglu_oai_int8_quant_masked, "Masked BF16-rounded OAI row INT8 quantization with zero/unit padding");
 m.def("swiglu_oai_fp8_quant_contiguous", &at::native::swiglu_oai_fp8_quant_contiguous, "Contiguous BF16 OAI activation and per-token E4M3FN quantization", pybind11::arg("input"), pybind11::arg("alpha") = 1.702, pybind11::arg("beta") = 1.0, pybind11::arg("limit") = 7.0);

 m.def("engram_hash_ids_hcu", &at::native::engram_hash_ids_hcu,
       "HCU Engram EXTEND predecessor lookup and exact INT64 hashing");

 m.def("hc_mix_stats_sinkhorn_bf16x3", &at::native::hc_mix_stats_sinkhorn_bf16x3,
       py::arg("x"), py::arg("weight_parts"), py::arg("scale"), py::arg("base"),
       py::arg("sinkhorn_iters"), py::arg("rms_eps"), py::arg("hc_eps"), py::arg("hc_mult")=4);
 m.def("hc_mix_stats_sinkhorn_deepgemm", &at::native::hc_mix_stats_sinkhorn_deepgemm,
       py::arg("x_flat"), py::arg("weight_parts"), py::arg("hc_scale"), py::arg("hc_base"),
       py::arg("sinkhorn_iters"), py::arg("rms_eps"), py::arg("hc_eps"), py::arg("hc_mult")=4);
 m.def("hc_mix_stats", &at::native::hc_mix_stats,
       "RMS-normalized projection, or None when unsupported", py::arg("x"),
       py::arg("weight"), py::arg("eps") = 1e-6);
 m.def("dpskv4_hc_head_fused", &at::native::dpskv4_hc_head_fused, "dpskv4_hc_head_fused");
m.def("fuse_silu_and_mul_ep",&at::native::fuse_silu_and_mul_ep,"fuse_silu_and_mul_ep");

  m.def("interleave_rows_", &lightop::cp_output::interleave_rows_,
        py::arg("gathered"), py::arg("cp_size"), py::arg("cycle_leaders"));
  m.def("make_interleave_plan_", &lightop::cp_output::make_interleave_plan_,
        py::arg("cycle_leaders"), py::arg("rows"), py::arg("cp_size"));

}

#endif  // focused extension or complete lightop.op
#endif  // focused mHC/CP extension or existing exports
