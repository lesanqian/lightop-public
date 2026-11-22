// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Public interfaces implemented by csrc/kernels/rmsnorm/*.cu.
// This is the sole binding owner in both full and selective builds.
// Preserve the existing Python signatures, defaults and capability markers.
#include <torch/extension.h>

namespace at::native {

Tensor rmsnorm_forward_autograd(const Tensor& x,const Tensor& weight,double eps,bool training);

std::tuple<Tensor,Tensor>rmsnorm_forward(const Tensor& self,const Tensor& weight,c10::optional<Tensor>& output_inplace,double eps,bool training);

std::tuple<Tensor,Tensor>rmsnorm_backward(const Tensor& grad, const Tensor& X,const Tensor& rstd,const Tensor& gamma);

void fused_rms_norm_contiguous(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);

void fused_rms_norm_rope_contiguous(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit, bool flashmla_fp8_layout, float kv_quant_epsilon, const c10::optional<torch::Tensor>& kv_smooth_scale);

void fused_rms_norm_rope_contiguous_q_cat(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed, torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, torch::Tensor& q_nope, torch::Tensor& q_cat, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit, bool flashmla_fp8_layout, float kv_quant_epsilon, const c10::optional<torch::Tensor>& kv_smooth_scale);

void fuse_rmsnorm_rope_quant_qkv(torch::Tensor& positions, torch::Tensor& q_c, torch::Tensor& q_pe, torch::Tensor& q_quant, torch::Tensor& q_scale, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit);

void gammarms_rotary_embedding_fuse_with_kv_store(
    Tensor& positions,
    Tensor& query,
    Tensor& key,
    Tensor& v,
    int64_t head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    std::optional<Tensor>& weight_q,
    std::optional<Tensor>& weight_k,
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    Tensor& k_buffer,
    Tensor& v_buffer,
    Tensor& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    double epsilon,
    bool use_fp8_e4m3,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    bool is_interleaved);

void rms_rotary_embedding_fuse_with_kv_store(
    Tensor& positions,
    Tensor& query,
    Tensor& key,
    Tensor& v,
    int64_t head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    std::optional<Tensor>& weight_q,
    std::optional<Tensor>& weight_k,
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    Tensor& k_buffer,
    Tensor& v_buffer,
    Tensor& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    double epsilon,
    bool use_fp8_e4m3);

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

void fuse_fill_rms_x2_concat(torch::Tensor& hidden_states_fuse, torch::Tensor& positions, torch::Tensor& inputs_embeds, torch::Tensor& previous_hidden_states, torch::Tensor& weight_inputs_embeds, torch::Tensor& weight_previous_hidden_states, float epsilon);

Tensor fuse_rn_dropout_forward_autograd(const Tensor& x,const Tensor& weight,double eps,double rate,bool training);

Tensor fuse_rn_add_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,bool training,bool inplace);

Tensor fuse_rn_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,double rate,bool training);

Tensor fuse_rn_add_dropout_bias_forward_autograd(const Tensor& x,const Tensor& y,const Tensor &bias, const Tensor& weight,double eps,double rate,bool training);

void rms_norm_dynamic_per_token_quant(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input,double limit);

void rms_norm_dynamic_per_token_quant_sglang(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input,double limit,std::optional<Tensor> smooth_scale,std::optional<Tensor> norm_output,bool clamp_norm_output,std::optional<Tensor> residual_output,std::optional<Tensor> clamped_output);

void mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(Tensor& q_out,Tensor& packed_input,const Tensor& q_weight,const Tensor& kv_weight,
                                                       Tensor q_scales,double q_epsilon,double kv_epsilon);

void rms_norm_fp8_quant(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input, int fp8type, std::optional<Tensor> norm_output);

void rms_rotary_embedding_fuse(Tensor &positions,Tensor &query,Tensor &key,int64_t head_size,Tensor &cos_sin_cache, bool is_neox,Tensor weight_q,Tensor weight_k,c10::optional<Tensor> residual_q,c10::optional<Tensor> residual_k,double epsilon);

void fuse_rms_mrope_cuda(Tensor &q, Tensor &k, const Tensor &cos, const Tensor &sin, const std::vector<int>& mrope_section, int head_size, int rotary_dim,bool is_interleaved,Tensor weight_q,Tensor weight_k,c10::optional<Tensor> residual_q,c10::optional<Tensor> residual_k,double epsilon);

void rms_norm_opt(torch::Tensor& out,torch::Tensor& input,torch::Tensor& weight,double epsilon,double limit);

void rms_norm_no_weight(
    torch::optional<torch::Tensor> out,
    torch::Tensor& input,
    torch::optional<torch::Tensor> weight,
    double epsilon);

void fused_add_rms_norm_opt(torch::Tensor& input,torch::Tensor& residual, torch::Tensor& weight,double epsilon,double limit,bool use_aligned_loads = false);

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

std::vector<torch::Tensor> minimax_qk_norm_rope(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,int64_t,bool,int64_t);

}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp RMSNorm operators";
  m.attr("MLA_ROPE_SUPPORTED_Q_HEAD_STRIDES") = py::make_tuple(192, 256);
  m.attr("RMS_NORM_FP8_QUANT_VERSION") = 2;

  m.def("rmsnorm_forward_autograd", &at::native::rmsnorm_forward_autograd, "rmsnorm forward autograd");

  m.def("rmsnorm_forward", &at::native::rmsnorm_forward, "rmsnorm forward");

  m.def("rmsnorm_backward", &at::native::rmsnorm_backward, "rmsnorm backward");

  m.def("fused_rms_norm_contiguous", &at::native::fused_rms_norm_contiguous, "fused_rms_norm_contiguous");

  m.def(
      "fused_rms_norm_rope_contiguous",
      &at::native::fused_rms_norm_rope_contiguous,
      "fused_rms_norm_rope_contiguous",
      py::arg("positions"),
      py::arg("q_pe"),
      py::arg("k_pe"),
      py::arg("kv_c"),
      py::arg("kv_c_normed"),
      py::arg("weight"),
      py::arg("cos_sin_cache"),
      py::arg("slot_mapping"),
      py::arg("kv_cache"),
      py::arg("kv_cache_dtype_str"),
      py::arg("kv_cache_scale"),
      py::arg("is_neox_style"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0,
      py::arg("flashmla_fp8_layout") = false,
      py::arg("kv_quant_epsilon") = 1e-6f,
      py::arg("kv_smooth_scale") = py::none());

  m.def(
      "fused_rms_norm_rope_contiguous_q_cat",
      &at::native::fused_rms_norm_rope_contiguous_q_cat,
      "fused_rms_norm_rope_contiguous_q_cat",
      py::arg("positions"),
      py::arg("q_pe"),
      py::arg("k_pe"),
      py::arg("kv_c"),
      py::arg("kv_c_normed"),
      py::arg("weight"),
      py::arg("cos_sin_cache"),
      py::arg("slot_mapping"),
      py::arg("kv_cache"),
      py::arg("q_nope"),
      py::arg("q_cat"),
      py::arg("kv_cache_dtype_str"),
      py::arg("kv_cache_scale"),
      py::arg("is_neox_style"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0,
      py::arg("flashmla_fp8_layout") = false,
      py::arg("kv_quant_epsilon") = 1e-6f,
      py::arg("kv_smooth_scale") = py::none());

  m.def(
      "fuse_rmsnorm_rope_quant_qkv",
      &at::native::fuse_rmsnorm_rope_quant_qkv,
      "fuse_rmsnorm_rope_quant_qkv",
      py::arg("positions"),
      py::arg("q_c"),
      py::arg("q_pe"),
      py::arg("q_quant"),
      py::arg("q_scale"),
      py::arg("k_pe"),
      py::arg("kv_c"),
      py::arg("kv_c_normed"),
      py::arg("weight"),
      py::arg("cos_sin_cache"),
      py::arg("slot_mapping"),
      py::arg("kv_cache"),
      py::arg("kv_cache_dtype_str"),
      py::arg("kv_cache_scale"),
      py::arg("is_neox_style"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0);

  m.def("gammarms_rotary_embedding_fuse_with_kv_store", &at::native::gammarms_rotary_embedding_fuse_with_kv_store, "gammarms_rotary_embedding_fuse_with_kv_store");

  m.def("rms_rotary_embedding_fuse_with_kv_store", &at::native::rms_rotary_embedding_fuse_with_kv_store, "rms_rotary_embedding_fuse_with_kv_store");

  m.def("split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant", &at::native::split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant, "split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant");

  m.def("fuse_fill_rms_x2_concat", &at::native::fuse_fill_rms_x2_concat, "fuse_fill_rms_x2_concat");

  m.def("rn_dropout_forward_autograd", &at::native::fuse_rn_dropout_forward_autograd, "fused_rn_dropout forward");

  m.def("rn_add_forward_autograd", &at::native::fuse_rn_add_forward_autograd, "fused_rn_add forward");

  m.def("rn_add_dropout_forward_autograd", &at::native::fuse_rn_add_dropout_forward_autograd, "fused_rn_add_dropout forward");

  m.def("rn_add_dropout_bias_forward_autograd", &at::native::fuse_rn_add_dropout_bias_forward_autograd, "fused_rn_add_dropout_bias forward");

  m.def(
      "rms_norm_dynamic_per_token_quant",
      &at::native::rms_norm_dynamic_per_token_quant,
      "rms_norm_dynamic_per_token_quant",
      py::arg("output"),
      py::arg("input"),
      py::arg("weight"),
      py::arg("scales"),
      py::arg("epsilon"),
      py::arg("residual"),
      py::arg("update_input"),
      py::arg("limit") = -1.0);

  m.def(
      "rms_norm_dynamic_per_token_quant_sglang",
      &at::native::rms_norm_dynamic_per_token_quant_sglang,
      "rms_norm_dynamic_per_token_quant_sglang",
      py::arg("output"),
      py::arg("input"),
      py::arg("weight"),
      py::arg("scales"),
      py::arg("epsilon"),
      py::arg("residual") = py::none(),
      py::arg("update_input") = true,
      py::arg("limit") = -1.0,
      py::arg("smooth_scale") = py::none(),
      py::arg("norm_output") = py::none(),
      py::arg("clamp_norm_output") = true,
      py::arg("residual_output") = py::none(),
      py::arg("clamped_output") = py::none());

  m.def("mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang",&at::native::mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang,
        "mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang");

  m.def("rms_norm_fp8_quant", &at::native::rms_norm_fp8_quant,
        py::arg("out"), py::arg("input"), py::arg("weight"), py::arg("scales"),
        py::arg("epsilon"), py::arg("residual"), py::arg("update_input"),
        py::arg("fp8type"), py::arg("norm_output") = py::none());

  m.def("rms_rotary_embedding_fuse", &at::native::rms_rotary_embedding_fuse,"rms_rotary_embedding_fuse");

  m.def("fuse_rms_mrope_cuda", &at::native::fuse_rms_mrope_cuda,"fuse_rms_mrope_cuda");

  m.def(
      "rms_norm_opt",
      &at::native::rms_norm_opt,
      "rms_norm_opt",
      py::arg("out"),
      py::arg("input"),
      py::arg("weight"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0);

  m.def("rms_norm_no_weight", &at::native::rms_norm_no_weight, "rms_norm_no_weight");

  m.def(
      "fused_add_rms_norm_opt",
      &at::native::fused_add_rms_norm_opt,
      "fused_add_rms_norm_opt",
      py::arg("input"),
      py::arg("residual"),
      py::arg("weight"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0,
      py::arg("use_aligned_loads") = false);

  m.def("fused_rms_norm_rope_int8quant_contiguous", &at::native::fused_rms_norm_rope_int8quant_contiguous, "fused_rms_norm_rope_int8quant_contiguous");

  m.def("rms_norm_smooth_per_token_dynamic_quant", &at::native::rms_norm_smooth_per_token_dynamic_quant, "rms_norm_smooth_per_token_dynamic_quant");

  m.def("fuse_qkv_head_rms_norm", &at::native::fuse_qkv_head_rms_norm, "fuse_qkv_head_rms_norm");

  m.def("gemma_fused_add_rmsnorm", &at::native::gemma_fused_add_rmsnorm, "gemma_fused_add_rmsnorm");

  m.def("gemma_rmsnorm", &at::native::gemma_rmsnorm, "gemma_rmsnorm");

  m.def("gemma_rms_norm_fp8_quant", &at::native::gemma_rms_norm_fp8_quant, "gemma_rms_norm_fp8_quant");

  m.def("qkv_split_rms_rotary_embedding_fuse", &at::native::qkv_split_rms_rotary_embedding_fuse, "qkv_split_rms_rotary_embedding_fuse");

  m.def("gemma_qkv_split_rms_rotary_embedding_fuse", &at::native::gemma_qkv_split_rms_rotary_embedding_fuse, "gemma_qkv_split_rms_rotary_embedding_fuse");

  m.def("mimo_v2_split_rope_vscale_kv_store", &at::native::mimo_v2_split_rope_vscale_kv_store, "mimo_v2_split_rope_vscale_kv_store");

  m.def("minimax_qk_norm_rope", &at::native::minimax_qk_norm_rope);
}
