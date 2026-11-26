// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// HCU MMAC 统一封装（gfx928 / gfx936 / gfx938 / gfx92a / gfx946）
//
// 三个指令族，按 DTK 的 target-feature 划分：
//
//   gfx92a        MMOP5  ：无 lit/lts 参数；INT8 操作数为 2×32bit(<2 x int>)，
//                          必须走 _i8_clamp 变体（int64 版 _i32_16x16x32_* 在 92a 不存在）
//   gfx938/gfx946 MMOP4+ ：需显式 lit/lts；INT8 操作数同为 <2 x int>，走 _lit_clamp_lts
//                          （lit/clamp/lts 全 false，与仓库其它 946 代码保持一致）
//   gfx928/gfx936 MMOP2  ：INT8 为 packed 64bit(int64_t)，走 _i32_16x16x32i8
//
// 一律使用 __builtin_hcu_* 接口。禁止再使用 __builtin_amdgcn_mmac_*：
// 它们在 gfx938 / gfx92a / gfx946 上均不可编译（且 hcu builtin 手册未收录），
// 详见 /opt/dtk/llvm/include/clang/Basic/BuiltinsLingShu.def 的 target-feature 谓词。

#if defined(__gfx92a__)

using hcu_mmac_intx2 = int __attribute__((ext_vector_type(2)));

#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16((a), (b), (c))
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16((a), (b), (c))
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32((a), (b), (c))
// 92a 没有 int64 形参的 INT8 MMAC，只能 bitcast 成 <2 x int> 走 clamp 变体
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_clamp( \
        __builtin_bit_cast(hcu_mmac_intx2, (a)), \
        __builtin_bit_cast(hcu_mmac_intx2, (b)), (c), 0)

#elif defined(__gfx938__) || defined(__gfx946__)

using hcu_mmac_intx2 = int __attribute__((ext_vector_type(2)));

#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts((a), (b), (c), false, false)
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts((a), (b), (c), false, false)
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts((a), (b), (c), false, false)
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts( \
        __builtin_bit_cast(hcu_mmac_intx2, (a)), \
        __builtin_bit_cast(hcu_mmac_intx2, (b)), (c), false, false, false)

#else  // gfx928 / gfx936（MMOP2）

#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16((a), (b), (c))
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16((a), (b), (c))
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32((a), (b), (c))
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32i8((a), (b), (c))

#endif

#if defined(__gfx946__)
// ============================================================================
// gfx946（少伯）原生 UE8M0 块缩放 MMAC 封装（ISA v1.8 §2.5.9 / §3.14.4）
//
// 1. Scale Buffer 硬件规格：每 CU 8 KB = 32 slots（1 slot = 256 B = 16 rows × 16 cols），
//    每行存 16 个 8-bit E8M0 缩放因子（对应 16×16×32 MMAC 的 16 行 A 或 16 列 B）。
// 2. M0 寄存器寻址约定（wave 私有）：
//      - M0[8:0]   = ScaleA_row（A 块在 Scale Buffer 中的基行号，8-row/128B 对齐）
//      - M0[24:16] = ScaleB_row（B 块在 Scale Buffer 中的基行号，8-row/128B 对齐）
//    实际读取行号为 ScaleA_row + scale_a_offset、ScaleB_row + scale_b_offset。
// 3. Data Hazard（ISA v1.8 §4.2.5）：
//      - SALU 写 M0 后到下游读取 M0 需要 1 个 wait state，故显式插入 `s_nop 0`。
//      - 注意：`ds_read_matrix_[trans_]format` 会覆写 M0 作为 LDS 地址指针，因此
//        HCU_SET_SCALE_M0 必须在同拍所有 ds_read_matrix_* 发射完毕之后、
//        v_mmac_scale_* 发射之前调用！
// ============================================================================
#define HCU_SET_SCALE_M0(row_a, row_b)                                          \
    do {                                                                        \
        int _hcu_scale_m0 = __builtin_amdgcn_readfirstlane(                     \
            (((row_b) & 0x1FF) << 16) | ((row_a) & 0x1FF));                     \
        asm volatile("s_mov_b32 m0, %0\n\ts_nop 0"                              \
                     :: "s"(_hcu_scale_m0) : "memory");                         \
    } while (0)

// 硬件块缩放矩阵乘加：D.f32 = (A.fp8 * 2^(scaleA-127)) x (B.fp8 * 2^(scaleB-127)) + C.f32
// - RealABTypeId = 0 对应 FP8_FP8（E4M3 x E4M3）
// - lit = true, lts = false 时每 lane 持有同一行连续 4 列（row = lane&15, col = (lane>>4)*4+[0..3]），
//   可直接合并向量写回，免 Epilogue LDS 转置
// - off_a / off_b 为编译期常量（范围 [0, 31]），编码进指令 SOH.Scale_offset[9:0]
#define HCU_MMAC_SCALE_F32_FP8_FP8(a, b, c, lit, lts, off_a, off_b)             \
    __builtin_hcu_mmac_scale_f32_16x16x32_f8f6f4_lit_lts(                       \
        (a), (b), (c), 0, (lit), (lts), (off_a), (off_b))
#endif
