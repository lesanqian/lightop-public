// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// ===========================================================================
// warp / wave size 跨 torch 版本兼容层
// ===========================================================================
//
// 背景
// ----
// torch < 2.12（ROCm 6.x / DTK 25.x）
//   C10_WARP_SIZE 在 host 与 device 两侧都是编译期常量（ROCm 下即 hip 头文件里的
//   constexpr warpSize，host 侧固定 64），因此可以直接用于：
//     constexpr 变量、static_assert、__shared__ 数组维度、模板默认参数、__launch_bounds__。
//
// torch >= 2.12（ROCm 7.x / DTK 26.04）
//   ROCm 7 起 warpSize 不再是 constexpr，torch 把 C10_WARP_SIZE 改成了"运行时取值"
//   （C10_WARP_SIZE_INTERNAL()，host 侧等价于 at::cuda::warp_size()），同时新增两个
//   编译期上/下界宏：
//     C10_WARP_SIZE_LOWER_BOUND / C10_WARP_SIZE_UPPER_BOUND
//   此时在 host 侧继续把 C10_WARP_SIZE 当常量用会直接编译失败（典型报错：
//   "constexpr int share_size = block_size / C10_WARP_SIZE" 或
//   "static assertion failed ... 64 <= 32"）。
//
// 本文件的做法（兼容，而不是直接替换）
// ------------------------------------
// 1) 对旧版 torch 补齐新接口：把 C10_WARP_SIZE_UPPER_BOUND / _LOWER_BOUND 别名到
//    旧版的 C10_WARP_SIZE。旧版 C10_WARP_SIZE 本身就是唯一取值，所以别名之后
//    数值、共享内存尺寸、断言结果与改造前完全一致 —— 旧行为被完整保留。
// 2) 对新版 torch 不覆盖任何 torch 定义，直接沿用官方编译期上下界。
// 3) 运行期 warp size 统一走 lightop::compat::warp_size()：新版 torch 调
//    at::cuda::warp_size()，旧版 torch 回落到常量 C10_WARP_SIZE。
// 4) 额外提供 C10_WARP_SIZE_STATIC 别名，兼容历史上
//    "#define C10_WARP_SIZE C10_WARP_SIZE_STATIC" 的写法。
//
// 使用规则
// --------
//   * 编译期用途（static_assert / __shared__ 维度 / 模板默认参数 / __launch_bounds__）
//     -> C10_WARP_SIZE_UPPER_BOUND
//        取"上界"：共享内存按最大 warp 数分配不会越界；gfx9(wave64) / gfx10+(wave32)
//        在 device 编译期分别取 64 / 32，与改造前按实际 warp size 分配完全一致。
//   * 运行期用途（kernel launch 的 block 尺寸等动态配置）
//     -> lightop::compat::warp_size()
//   * device 代码内部的行列换算（threadIdx.x % C10_WARP_SIZE 等）
//     -> 继续使用 C10_WARP_SIZE，新旧 torch 在 device 侧都可用。
// ===========================================================================
#pragma once

#include <c10/macros/Macros.h>

// ---------------------------------------------------------------------------
// 1. 版本探测：torch 是否已经提供运行时 warp size 接口
//    （新版 torch 一定带 *_UPPER_BOUND / *_LOWER_BOUND，旧版一定没有）
// ---------------------------------------------------------------------------
#if defined(C10_WARP_SIZE_UPPER_BOUND) && defined(C10_WARP_SIZE_LOWER_BOUND)
#define LIGHTOP_TORCH_HAS_RUNTIME_WARP_SIZE 1
#else
#define LIGHTOP_TORCH_HAS_RUNTIME_WARP_SIZE 0
#endif

// ---------------------------------------------------------------------------
// 2. 旧版 torch 补齐新版接口（保留旧接口 + 兼容新接口）
//    注意：必须放在探测之后，否则会误判版本。
// ---------------------------------------------------------------------------
#if !defined(C10_WARP_SIZE_UPPER_BOUND)
// 旧版：C10_WARP_SIZE 即编译期常量，上下界同值
#define C10_WARP_SIZE_UPPER_BOUND C10_WARP_SIZE
#endif

#if !defined(C10_WARP_SIZE_LOWER_BOUND)
#define C10_WARP_SIZE_LOWER_BOUND C10_WARP_SIZE
#endif

// 历史兼容：旧代码里的 C10_WARP_SIZE_STATIC（DTK 侧部分 torch 构建会自带该常量，
// 这里只在缺失时补齐，避免覆盖 torch 自己的定义）。
#if !defined(C10_WARP_SIZE_STATIC)
#define C10_WARP_SIZE_STATIC C10_WARP_SIZE_UPPER_BOUND
#endif

namespace lightop {
namespace compat {

// -------------------------------------------------------------------------
// 运行期 warp / wave size（DCU gfx9 系列 = 64，NVIDIA = 32）
// 旧版 torch：C10_WARP_SIZE 在 host 侧就是常量，直接返回等价于改造前的
//             "NumWarps * C10_WARP_SIZE" 写法。
// 新版 torch：走官方运行时查询，保证与实际设备一致。
// -------------------------------------------------------------------------
inline int warp_size() {
#if LIGHTOP_TORCH_HAS_RUNTIME_WARP_SIZE && defined(USE_ROCM)
  return at::cuda::warp_size();
#else
  return C10_WARP_SIZE;
#endif
}

}  // namespace compat
}  // namespace lightop
