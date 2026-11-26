# 07 架构收口审计与工业化加固方案（Items 1–6）

> **背景**：在 `v0.15.0` 初步落地 `ir_core.py`、`CompileContext`、`RewriteDriver` 与 `AetherPipeline` 后，通过全仓源码扫描与真实模型（Qwen2 / DeepSeek / LLaMA）实测，发现统一框架在覆盖率、校验严格度、数值真实性与后端落地方面仍有 6 类具体缺口。本文档记录审计证据与对应收口实现。

---

## 一、审计发现与收口目标

| 编号 | 缺口 | 收口方案 |
|---|---|---|
| **1. 框架收口** | `AetherPipeline` 缺少 FlashAttention/AddRMSNorm/HorizontalFusion/GlobalLayout；`AetherCompiler` 与 `AetherPipeline` 双入口割裂；各 Pass 各自硬编码 `128/256/512` FLOPs/cycle；`placement_spec` 与 `attrs` 双来源；`CSE`/`GraphScheduler` 未遵守 `OpEffect` | 在 `CompileContext` 内置唯一 `UnifiedHardwareCostModel`；`RewriteDriver` 补齐全部融合 Pattern 成为旧 `OperatorFusionPass` 的严格超集；`AetherPipeline` 纳入 `GlobalLayout`/`HorizontalFusion`/`FlowDispatch`/`AMP`；`AetherCompiler` 支持统一上下文与每 Pass 校验；`CSE`/`GraphScheduler` 严格遵守 `OpEffect` |
| **2. 强 Verifier + 随机差分测试** | `GraphVerifier` 未调用 `infer_shapes`、未校验 `dtype`、未递归检查 `Region`；缺少随机图差分测试 | `GraphVerifier` 强制校验算子形状契约、dtype 一致性、并递归校验嵌套 `Region`；新增 `tests/test_differential_fuzz.py` 生成 30+ 随机 DAG 与多套 Pass 流水线差分对拍 |
| **3. 文本 IR 无损往返** | `to_aether_ir()` 丢弃 `Region`（`call`/`cond`/`fused_region` 解析后报 `KeyError`）与 `initial_value` 常量数值 | 扩展 `dump_aether_ir` / `parse_aether_ir` 支持嵌套 `region @name(...) { ... }` 与 base64 编码的 IEEE-754 常量权重往返，解析后可直接执行且逐位一致 |
| **4. 数值真实化（FP16/BF16/FP8/AMP/KV 量化）** | `cast` 为恒等函数无真实舍入；缺少 FP8、AMP 与 KV Cache 量化 | 实现 `numeric_types.py`（IEEE FP16、BF16 RNE、OCP FP8 E4M3/E5M2、INT8 量化）接入 `cast`；新增 `MixedPrecisionPass` (`amp`) 与 `KVQuantizationPass` (`kv-quant`) 并实测精度误差 |
| **5. 全算子 `emit_c` + Native C JIT** | 15 个新 LLM 算子无 `emit_c`，只能跑 Python 模拟器无法走 `gcc` Native JIT | 为全部新 LLM 算子（`embedding`/`reshape`/`permute`/`concat`/`slice`/`gather`/`topk`/`attention`/`paged_attention`/`paged_kv_append`/`moe_router`/`moe_combine`/`all_reduce`/`fused_region`/`call`/`cond`）补齐自包含 C99 代码发射，并在 `gcc` Native C JIT 下与模拟器逐位对拍 |
| **6. `feat/llvm-mlir` 真实 MLIR 方言起步** | `feat/llvm-mlir` 尚无 `.td` / `.cpp` / `CMakeLists.txt` 文件 | 在 `feat/llvm-mlir` 落地标准 Out-of-Tree MLIR 工程 `mlir/`（`AetherGraph` / `AetherMem` / `AetherPlacement` 三大方言 TableGen `.td`、C++ Pass、`aether-opt` 工具与 `.mlir` 测试），并在 Python 侧提供 `compiler/backend/mlir_exporter.py` 导出对应方言文本 |
