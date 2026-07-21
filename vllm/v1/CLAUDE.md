# V1 引擎 CLAUDE.md

本文件对 `vllm/v1/` 目录提供指导。阅读时请结合项目级 [CLAUDE.md](../../CLAUDE.md)。

## 架构总览

V1 采用 **调度器-核心-执行器（Scheduler-Core-Worker）** 三层架构：

```
API 层 (entrypoints/openai)
    │ ZMQ
EngineCore (core.py)          ← 主循环：调度 → 执行 → 采样
    ├── Scheduler              ← 决定每步跑哪些请求
    ├── KVCacheManager         ← KV 缓存块管理
    └── ModelRunner            ← 执行模型前向传播
        └── Worker(s)          ← GPU/CPU 进程
```

关键设计决策：
- **ZMQ 通信**：API 层与 EngineCore 通过 ZMQ 解耦，不在同一个 Python 进程中
- **异步调度**：`AsyncScheduler` 使调度与模型执行可以重叠（流水线）
- **block 级别 KV 缓存**：基于 PagedAttention 的块管理，支持前缀缓存和跨请求共享

## 模块协作链路

一个请求的完整生命周期：

```
1. API 层接收请求
     ↓
2. coordinator.py 将请求发送给 EngineCore（ZMQ PUSH/PULL）
     ↓
3. core.py: EngineCore.step()
   ├── scheduler.schedule()        → 选出本轮要执行的请求
   ├── model_runner.execute_model() → 执行模型前向
   └── sampler.sample()            → 采样得到 token
     ↓
4. output_processor.py 处理引擎输出
     ↓
5. detokenizer.py 增量解码 token → 文本
     ↓
6. coordinator 将结果推送给 API 层
```

## 各子目录关键文件

### engine/ — 引擎编排层

| 文件 | 职责 | 注意 |
|---|---|---|
| `core.py` | `EngineCore` 主循环，最大文件之一 | 改这里会影响所有请求链路 |
| `coordinator.py` | 异步前端 ↔ EngineCore 桥接 | 管理请求队列、生命周期 |
| `core_client.py` | `EngineCoreClient` 封装 ZMQ 通信 | ZMQ socket 配置在这里 |
| `llm_engine.py` | `LLMEngine` 对外 API | V1 的公共入口 |
| `async_llm.py` | `AsyncLLM` 异步引擎 | 在线服务使用 |
| `input_processor.py` | 请求预处理（tokenize、多模态） | embedding/flashinfer 模式在此分流 |
| `output_processor.py` | 引擎输出后处理 | logprobs、stop reason 等 |
| `detokenizer.py` | 增量解码 | 支持多种 tokenizer 后端 |

### core/ — 调度器与 KV 缓存

| 文件 | 职责 | 注意 |
|---|---|---|
| `sched/scheduler.py` | `Scheduler` 核心调度逻辑 | prefill/decode 分离，前缀缓存 |
| `sched/async_scheduler.py` | 异步调度 wrapper | 调度与执行重叠的关键 |
| `sched/output.py` | `SchedulerOutput` 数据结构 | 新增调度信息需修改此处 |
| `sched/interface.py` | `SchedulerInterface` 抽象 | 定义了调度器的公共契约 |
| `kv_cache_manager.py` | KV 缓存块分配/淘汰/前缀匹配 | PagedAttention 核心 |
| `kv_cache_coordinator.py` | `HybridKV` 异构缓存协调 | 多种 attention 类型并存时使用 |
| `block_pool.py` | 块内存池 | 底层内存管理 |

### worker/ — 设备端模型执行

| 文件 | 职责 | 注意 |
|---|---|---|
| `gpu_model_runner.py` | GPU 模型执行器，**仓库最大单文件** | forward、KV cache、attention metadata |
| `gpu_worker.py` | GPU worker 进程 | 持有 model runner |
| `gpu_input_batch.py` | GPU 输入批次数据结构 | 输入准备和批次组装 |
| `block_table.py` | KV 缓存块表 | block 级别寻址 |
| `ubatching.py` | 微批次调度 | 流水线并行的批次划分 |
| `cpu_model_runner.py` | CPU 模型执行器 | GPU 版本的 CPU 对应实现 |
| `workspace.py` | 工作空间内存管理 | 临时张量分配 |

### executor/ — 分布式执行

| 文件 | 职责 | 注意 |
|---|---|---|
| `abstract.py` | `Executor` 抽象基类 | 定义执行器接口 |
| `uniproc_executor.py` | 单进程执行器 | 单 GPU 场景 |
| `multiproc_executor.py` | 多进程执行器 | 基于 torch distributed |
| `ray_executor.py` | Ray 分布式执行器 | 多节点场景 |

### attention/ — Attention 后端选择

| 文件 | 职责 |
|---|---|
| `selector.py` | 根据模型架构 + 硬件选择 attention 后端 |
| `backends/` | FlashAttention、FlashInfer、FlashMLA、Triton 等实现 |

新增 attention 后端需要在 `backends/` 添加实现，并在 `selector.py` 注册。

### sample/ — 采样

| 文件 | 职责 |
|---|---|
| `sampler.py` | Token 采样（top-k、top-p、temperature、min-p 等） |
| `logits_processor/` | Logits 处理器（repetition penalty、frequency penalty 等） |
| `rejection_sampler.py` | 投机解码的拒绝采样 |

### structured_output/ — 结构化生成

基于 xgrammar、guidance、outlines、LM Format Enforcer 等后端实现约束生成。`backend_xgrammar.py` 是当前主力后端。

### spec_decode/ — 投机解码

EAGLE、DFlash、Medusa、n-gram、后缀解码等投机解码策略。每种策略实现 `Proposer`，生成候选 token 由 `rejection_sampler.py` 验证。

## 调试技巧

- **开启调度日志**：`VLLM_LOGGING_LEVEL=DEBUG` 可以看到每个 step 调度了哪些请求
- **追踪引擎输出**：`VLLM_TRACE_FUNCTION=1` 打印函数调用栈
- **ZMQ 通信调试**：检查 `core_client.py` 中的 socket 超时配置
- **GPU memory 问题**：关注 `block_pool.py` 和 `kv_cache_manager.py` 的内存分配逻辑
