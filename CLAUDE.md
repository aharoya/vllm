# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

vLLM 是一个高吞吐量、低内存占用的 LLM 推理与服务引擎，支持 200+ 种模型架构，覆盖 NVIDIA/AMD/Intel GPU、CPU、TPU 及多种 AI 加速器。

## 常用命令

### 环境初始化（必选）

只做一件事：创建隔离的 Python 环境，然后把 vLLM 装进去。

**uv 方式：**

```bash
uv venv --python 3.12
source .venv/bin/activate
```

**conda 方式：**

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

### 安装 vLLM（必选）

将当前源码以可编辑模式（`-e`）安装到环境中，源码改动无需重新安装即可生效。

```bash
# 仅改 Python 代码时（最快，跳过 C++/CUDA 编译）：
VLLM_USE_PRECOMPILED=1 pip install -e .

# 涉及 C/C++ 改动时（需本地编译）：
pip install -e .
```

> uv 用户可用 `uv pip install -e . --torch-backend=auto`（`--torch-backend` 为 uv 特有参数）。

### 代码检查工具（可选）

安装 pre-commit 后，每次 `git commit` 会自动运行代码格式检查。在 fork 仓库上开发时通常不需要，可以跳过此步骤。

```bash
pip install -r requirements/lint.txt
pre-commit install
```

如已安装但想临时跳过检查：`git commit -n` 或 `git commit --no-verify`。

> 注意：`--torch-backend=auto` 是 `uv pip` 特有的参数，标准 `pip` 不支持。目标设备通过 `VLLM_TARGET_DEVICE` 环境变量控制（默认为 `cuda`），一般无需额外指定。

### 代码检查

```bash
# 仅检查已暂存文件：
pre-commit run

# 检查所有文件：
pre-commit run --all-files

# 单独运行某个 hook：
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

Python 代码行宽限制 88 字符。使用 Google 风格 docstring（`Args:`/`Returns:`/`Raises:`）。

### 测试

先安装测试依赖：
```bash
uv pip install -r requirements/test/cuda.in
```

运行测试：
```bash
# 单个测试文件：
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# 单个测试函数：
.venv/bin/python -m pytest tests/path/to/test_file.py::test_name -v -s

# 多 GPU 分布式测试（需加 --distributed）：
.venv/bin/python -m pytest tests/path/to/test_file.py -v -s --distributed

# V1 引擎测试：
.venv/bin/python -m pytest tests/v1/ -v -s

# 内核测试（需要 GPU）：
.venv/bin/python -m pytest tests/kernels/ -v -s

# 编译/融合 pass 测试：
.venv/bin/python -m pytest tests/compile/ -v -s
```

关键测试标志：
- `--distributed` — 启用多 GPU 分布式测试
- `-s` — 显示 stdout（调试时有用）
- `NUMBA_DISABLE_JIT=1` — 在 pytest 前设置，可加速部分 kernel 测试

### C++ / CUDA 构建

```bash
mkdir build && cd build
cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
cmake --build . --target install
```

### Rust 前端

```bash
cd rust
# 运行测试：
cargo nextest run

# 编译：
cargo build --release
```

## 架构

### 双引擎：V0（旧）与 V1（当前主力）

V1（`vllm/v1/`）是当前主力开发的新引擎。V0（`vllm/engine/`）为遗留引擎，仍在维护但不再新增特性。V1 采用 **调度器-核心-执行器（Scheduler-Core-Worker）** 架构，各模块职责如下：

**`vllm/v1/engine/`** — 引擎编排层：
- `core.py` — `EngineCore` 运行主循环：调度 → 模型执行 → 采样。通过 ZMQ 与 API 层通信。
- `core_client.py` — API 进程与 EngineCore 通信的客户端。
- `llm_engine.py` — `LLMEngine`（V1）对外公开 API，封装 EngineCore。
- `async_llm.py` — `AsyncLLM` 异步服务，通过 `coordinator.py` 与 EngineCore 协调。
- `coordinator.py` — 连接异步前端与 EngineCore，管理请求生命周期。
- `input_processor.py`、`output_processor.py` — 请求的前处理与后处理。

**`vllm/v1/core/`** — 调度器与 KV 缓存：
- `sched/scheduler.py` — `Scheduler` 决定每步调度哪些请求运行，管理 prefill/decode 调度、前缀缓存、投机解码、编码器缓存及 KV connector 集成。
- `sched/async_scheduler.py` — 调度器的异步封装，使调度与执行可重叠。
- `sched/interface.py` — 抽象 `SchedulerInterface` 接口。
- `sched/output.py` — 描述调度结果的 `SchedulerOutput`。
- `kv_cache_manager.py` — 管理 KV 缓存块（分配、淘汰、前缀缓存）。
- `kv_cache_coordinator.py` — `HybridKVCacheCoordinator` 管理异构 KV 缓存。
- `single_type_kv_cache_manager.py` — 管理单一类型的 KV 缓存。
- `block_pool.py` — 块内存池。

**`vllm/v1/worker/`** — 在各设备上执行模型推理：
- `gpu_model_runner.py` — GPU 模型执行器，整个仓库最大的单文件。执行模型前向传播，管理 KV 缓存、attention 元数据及输入准备。
- `gpu_worker.py` — GPU worker 进程，持有 model runner。
- `cpu_model_runner.py`、`cpu_worker.py` — CPU 版本对应实现。
- `gpu_input_batch.py` — 输入张量批次表示。
- `block_table.py` — KV 缓存块表管理。
- `ubatching.py` — 流水线并行的微批次调度。
- `workspace.py` — 工作空间内存管理。

**`vllm/v1/executor/`** — 分布式执行：
- `abstract.py` — 抽象 `Executor` 基类。
- `uniproc_executor.py` — 单进程执行（单 GPU 或 CPU）。
- `multiproc_executor.py` — 基于 torch distributed 的多进程执行。
- `ray_executor.py` — 基于 Ray 的分布式执行（V1，多节点场景包装 multiproc）。
- `ray_executor_v2.py` — 基于 compiled DAG 的替代 Ray 执行器。
- `ray_utils.py`、`ray_env_utils.py` — Ray 集群初始化与环境配置。

**`vllm/v1/attention/`** — Attention 后端：
- `backends/` — 各种实现：FlashAttention、FlashInfer、FlashMLA、Triton、ROCm 等。
- `selector.py` — 根据模型和硬件选择合适的 attention 后端。

**`vllm/v1/sample/`** — 采样与 logits 处理：
- `sampler.py` — Token 采样（top-k、top-p、temperature 等）。
- `logits_processor/` — Logits 处理器（惩罚项、min-p 等）。
- `rejection_sampler.py` — 投机解码的拒绝采样。

**`vllm/v1/structured_output/`** — 约束/结构化生成，支持 xgrammar、guidance、outlines、LM Format Enforcer 等后端。

**`vllm/v1/kv_offload/`** — GPU 内存不足时将 KV 缓存卸载至 CPU 或其他设备。

**`vllm/v1/spec_decode/`** — 投机解码：EAGLE、DFlash、Medusa、n-gram、后缀解码、draft 模型。

### Rust 前端（`rust/`）

一个替代 Python API 层的高性能前端实现，目前仍在早期开发阶段。

核心 crate：
- `server` — HTTP 服务器（基于 Axum），提供 API 服务
- `llm` — 主 LLM 接口
- `managed-engine` — 管理 Python 引擎进程的生命周期
- `engine-core-client` — 与 EngineCore 通信的客户端
- `tokenizer` — 分词器
- `parser` — 请求/响应解析器（基于 winnow，偏好声明式解析器）
- `chat` — 聊天模板渲染
- `bench` — 基准测试工具
- `text` — 文本处理工具
- `metrics` — Prometheus 指标
- `mock-engine` — 用于测试的 mock 引擎

需遵守 `rust/AGENTS.md` 中的 Rust 约定：使用快照测试（`expect-test`），优先使用声明式 winnow 解析器，错误报告使用 `thiserror-ext`，代码拆分为小模块。

### 模型执行器（`vllm/model_executor/`）

**`models/`** — 200+ 模型实现，每个文件通常实现一种模型架构（如 `llama.py`、`qwen2.py`）。每个文件导出通过 `ModelRegistry` 注册的模型类。

**`layers/`** — 可复用层组件：
- `linear.py` — 线性层，含量化支持
- `attention/` — Attention 层实现
- `rotary_embedding/` — RoPE 及其变体
- `fused_moe/` — 融合 MoE 内核（CUTLASS、TRTLLM 等）
- `quantization/` — 量化方案（FP8、INT8、INT4、GPTQ、AWQ、GGUF、compressed-tensors 等）
- `mamba/` — Mamba/状态空间模型层
- `pooler/` — 用于 embedding/分类模型的池化层
- `fusion/` — 层融合

**`model_loader/`** — 权重加载：默认（safetensors/PyTorch）、tensorizer（快速序列化）、bitsandbytes、分片状态、RunAI streamer。

**`layers/quantization/`** — 量化采用插件架构：每种方案提供自己的 config、linear method 以及可选的融合 MoE method。

### 入口点（`vllm/entrypoints/`）

- `openai/` — OpenAI 兼容 API 服务（chat completions、completions、embeddings）
- `anthropic/` — Anthropic Messages API 兼容
- `serve/` — 统一的 serving 引擎抽象层，被所有 API 协议共享
- `llm.py` — 离线 `LLM` 类，供编程式使用
- `cli/` — CLI 命令：`vllm serve`、`vllm bench`、`vllm chat` 等
- `grpc_server.py` — gRPC 服务支持
- `pooling/` — Pooling/embedding API 端点
- `speech_to_text/` — 语音转文字 API

### 配置（`vllm/config/`）

`VllmConfig`（`vllm.py`）是顶层配置，聚合了所有子配置：`ModelConfig`、`CacheConfig`、`ParallelConfig`、`CompilationConfig`、`AttentionConfig`、`DeviceConfig`、`LoadConfig`、`LoRAConfig`、`SchedulerConfig`、`SpeculativeConfig`、`MultiModalConfig`、`ObservabilityConfig`、`ProfilerConfig`、`StructuredOutputsConfig`、`OffloadConfig`、`KVTransferConfig`、`PoolerConfig`、`MambaConfig` 等。

### 分布式（`vllm/distributed/`）

- `parallel_state.py` — 进程组管理（张量并行/流水线并行/数据并行/专家并行/上下文并行）
- `device_communicators/` — GPU 通信后端：CUDA NCCL、custom all-reduce、PyNCCL、QuickReduce、FlashInfer all-reduce、SHM broadcast、Ray
- `kv_transfer/` — 分离式服务（disaggregated serving）：prefill 与 decode 实例间的 KV 缓存传输
- `ec_transfer/` — 弹性专家并行：动态 expert 放置和传输
- `eplb/` — 专家并行负载均衡
- `elastic_ep/` — 弹性专家并行
- `weight_transfer/` — 分离式服务的权重传输

### 编译优化（`vllm/compilation/`）

集成 `torch.compile` 和 CUDA Graph 以优化模型执行：
- `cuda_graph.py` — CUDA Graph 捕获与重放
- `piecewise_backend.py` — 逐层编译策略
- `partition_rules.py` — 用于分段编译的图划分
- `wrapper.py` — `torch.compile` 封装
- `passes/` — 图优化 pass（融合、清理、lowering）

### 其他关键模块

- `vllm/transformers_utils/` — HuggingFace 集成、tokenizer 封装、配置加载
- `vllm/multimodal/` — 多模态处理（图像、音频、视频）、媒体加载、注册表
- `vllm/lora/` — LoRA 适配器支持：权重加载、模型管理、算子
- `vllm/assets/` — 媒体资产处理（图像、音频、视频）
- `vllm/platforms/` — 平台检测：CUDA、ROCm、Intel XPU、TPU、CPU
- `vllm/plugins/` — 插件系统，动态注册 resolver、platform 等
- `vllm/inputs/` — 输入预处理
- `vllm/outputs.py` — 输出类型：RequestOutput、CompletionOutput、EmbeddingOutput 等
- `vllm/sampling_params.py` — 生成采样参数
- `vllm/sequence.py` — 序列状态管理
- `vllm/tokenizers/` — 自定义 tokenizer 支持
- `vllm/tool_parsers/` — 工具调用解析器
- `vllm/reasoning/` — 推理内容解析

### C++ / CUDA（`csrc/`）

性能关键操作的 native 扩展：
- `attention/` — 自定义 attention 内核
- `quantization/` — 量化内核（FP8、INT8、GPTQ、AWQ、MXFP）
- `moe/` — MoE 内核实现
- `custom_all_reduce.cuh`、`custom_quickreduce.cu` — GPU 集合通信
- `cumem_allocator.cpp` — CUDA 内存分配
- `torch_bindings.cpp` — PyTorch C++ 扩展绑定
- `cutlass_extensions/` — CUTLASS 模板特化
- `libtorch_stable/` — PyTorch C++ API 稳定性补丁

### 构建系统

- **CMake**（`CMakeLists.txt`、`cmake/`）— 构建 C++/CUDA 扩展。支持 CUDA、ROCm、Intel XPU 等目标平台。
- **setup.py** — Python 包构建，通过 `setuptools-rust` 编译 Rust 扩展，通过 CMake 集成编译 C++ 扩展。
- **pyproject.toml** — 项目元数据，构建依赖（torch 2.11.0），通过 setuptools-scm 管理版本号。
- **requirements/** — 依赖文件：`common.txt`、`cuda.txt`、`rocm.txt`、`cpu.txt`、`xpu.txt`、`tpu.txt`、`lint.txt`、`dev.txt`、`docs.txt`。
- **.pre-commit-config.yaml** — 代码检查：ruff（format + check）、typos、clang-format、markdownlint、actionlint、pip-compile。

### 环境变量（`vllm/envs.py`）

vLLM 大量使用环境变量进行配置（统一定义在 `vllm/envs.py`）。关键变量包括：
- `VLLM_USE_PRECOMPILED` — 使用预编译的 C++/CUDA 扩展
- `VLLM_TARGET_DEVICE` — 目标设备（cuda/rocm/xpu/cpu）
- `VLLM_ATTENTION_BACKEND` — 覆盖默认 attention 后端
- `VLLM_USE_FLASHINFER_SAMPLER` — 使用 FlashInfer 采样器
- `VLLM_CPU_KVCACHE_SPACE` — CPU KV 缓存空间（GB）
- `VLLM_USE_RAY_COMPILED_DAG` — 多节点使用 Ray compiled DAG

环境变量在 `vllm/__init__.py` 中通过 `vllm.env_override` 最先导入，确保在其他任何模块加载之前生效。

## 测试约定

- 测试目录与源码目录结构一一对应：`tests/v1/` 测试 `vllm/v1/`，`tests/kernels/` 测试 `vllm/kernels/` 等
- `tests/conftest.py` 提供共享 fixture（模型、分布式环境、HTTP 服务器）
- V1 测试位于 `tests/v1/`，子目录与源码一致（`v1/core/`、`v1/engine/`、`v1/worker/` 等）
- 编译/融合 pass 测试在 `tests/compile/`，有自己的 `conftest.py` 提供编译模型 fixture
- 模型测试位于 `tests/models/`，测试单个模型架构
- 分布式测试需要加 `--distributed` 标志；单 GPU 测试默认运行
- 测试辅助工具使用 `tests/vllm_test_utils/`
