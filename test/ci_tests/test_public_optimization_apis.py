import subprocess
import sys


def test_category_exports_preserve_aliases():
    import lightop
    from lightop import attention, kvcache, norm
    from lightop.attention.mqa import mqa_logits_topk_sampled_int8
    from lightop.fused_metadata import dsa_cp_prefill_metadata_plan

    assert attention.mqa_logits_topk_sampled_int8 is mqa_logits_topk_sampled_int8
    assert kvcache.dsa_cp_prefill_metadata_plan is dsa_cp_prefill_metadata_plan
    for name in ("rms_norm_dynamic_per_token_quant_sglang",
                 "mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang"):
        assert getattr(norm, name) is getattr(lightop, name)
        assert name in norm.__all__
    for category, names in (
        (attention, ("spark_mla_rope_cat_store", "indexer_q_rope_hadamard_quant",
                     "mla_q_absorb_rope_concat")),
        (kvcache, ("indexer_k_norm_rope_hadamard_store", "fill_swa_ring_mapping")),
    ):
        for name in names:
            assert getattr(category, name) is getattr(lightop.op, name)
            assert name in category.__all__
    for name in ("topk_transform_ragged_v2", "topk_transform_1024",
                 "fast_kpool_topk_transform_fused"):
        assert getattr(attention, name) is getattr(lightop, name)


def test_older_native_optional_symbols_do_not_break_imports():
    script = '''
import lightop
names = ("spark_mla_rope_cat_store", "indexer_q_rope_hadamard_quant",
         "mla_q_absorb_rope_concat", "indexer_k_norm_rope_hadamard_store",
         "fill_swa_ring_mapping")
for name in names:
    if hasattr(lightop.op, name):
        delattr(lightop.op, name)
from lightop import attention, kvcache, norm, quant
assert callable(attention.mqa_logits)
assert callable(norm.rms_norm)
assert callable(quant.per_token_dynamic_quant_int8)
assert not hasattr(kvcache, "fill_swa_ring_mapping")
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def test_phase3a_registry_and_module_routing():
    from pathlib import Path
    import lightop
    from lightop.core import MODULES, owner_binary

    expected_modules = {
        "rmsnorm",
        "mqa_logits",
        "marlin_w4a16",
        "marlin_w4a8",
        "marlin_w4fp8",
        "layernorm",
        "groupnorm",
        "moe",
        "asm_gemm",
        "lmslim_gemm",
        "kvcache",
        "quant",
        "attention_prep",
        "topk_sampling",
    }
    assert expected_modules.issubset(MODULES.keys())

    routed_symbols = (
        ("asm_gemm", "gemm_w8a8_smooth_asm"),
        ("asm_gemm", "moe_w8a8_asm"),
        ("lmslim_gemm", "_lmslimquant_rocblas_gemm_fp16"),
        ("lmslim_gemm", "gemm_awq_w4a16"),
        ("kvcache", "store_kv_cache_paged"),
        ("kvcache", "dsv4_pack_c4_topk_workspace"),
        ("kvcache", "PREFILL_GATHER_TRITON_NAN_COMPAT_VERSION"),
        ("quant", "per_token_quant_fp8"),
        ("quant", "PER_TOKEN_QUANT_FP8_VERSION"),
        ("quant", "per_token_quant_int8_out_supports_gate"),
        ("attention_prep", "rotary_embedding_deepseek_fuse"),
        ("attention_prep", "dsv41_build_swa_token_ids"),
        ("attention_prep", "ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI"),
        ("topk_sampling", "topk_transform_1024"),
        ("topk_sampling", "dsv41_ragged_topk_postprocess"),
        ("topk_sampling", "markov_gemv_add_argmax"),
    )
    for group, symbol in routed_symbols:
        assert hasattr(lightop.op, symbol), f"missing Phase 3A export: {symbol}"
        binary_name = Path(owner_binary(symbol)).name
        assert binary_name.startswith(f"module_{group}."), (
            f"{symbol} routed to {binary_name!r} instead of module_{group}"
        )


def test_selective_lmslim_gemm_import_without_rmsnorm():
    script = '''
import lightop
if hasattr(lightop.op, "rms_norm_dynamic_per_token_quant"):
    delattr(lightop.op, "rms_norm_dynamic_per_token_quant")
from lightop.gemm_ops import hipblaslt_w8a8_gemm, rocblas_scaled_mm_nn
assert callable(hipblaslt_w8a8_gemm)
assert callable(rocblas_scaled_mm_nn)
'''
    subprocess.run([sys.executable, "-c", script], check=True)
