# 测试 CLAUDE.md

本文件对 `tests/` 目录提供指导。阅读时请结合项目级 [CLAUDE.md](../CLAUDE.md)。

## 测试目录结构

测试目录与源码一一对应：

```
tests/
├── v1/                    ← 测试 vllm/v1/
│   ├── engine/            ← 引擎层测试
│   ├── core/              ← 调度器/KV Cache 测试
│   ├── e2e/               ← V1 端到端测试
│   ├── distributed/       ← V1 分布式测试
│   ├── attention/         ← Attention 后端测试
│   ├── sample/            ← 采样测试
│   └── ...
├── models/                ← 模型测试
│   ├── language/generation/  ← 文本生成模型
│   ├── multimodal/         ← 多模态模型
│   └── quantization/       ← 量化模型测试
├── kernels/               ← 内核测试（attention、core、moe、quantization）
├── compile/               ← 编译/fusion 测试
├── distributed/           ← 分布式通信测试
├── lora/                  ← LoRA 测试
├── entrypoints/           ← API 入口点测试
└── vllm_test_utils/       ← 测试工具包（独立，不导 vLLM 模块）
```

## 共享 Fixture（`tests/conftest.py`）

conftest 是整个测试套件的 fixture 注册中心：

### 核心 Runner Fixture

| Fixture | 作用 |
|---|---|
| `hf_runner` | 启动 HuggingFace 模型做对照验证（会话级别，复用模型） |
| `vllm_runner` | 启动 vLLM 实例（`seed=0`、`max_model_len=1024`、`block_size=16`） |

使用模式：
```python
with hf_runner(model_name) as hf_model:
    hf_output = hf_model.generate(prompts, max_tokens, **kwargs)
with vllm_runner(model_name, dtype=..., enforce_eager=...) as vllm_model:
    vllm_output = vllm_model.generate(prompts, max_tokens, **kwargs)

check_logprobs_close(hf_output, vllm_output, ...)
```

### 多模态资源 Fixture（会话级别，单例）

| Fixture | 类型 |
|---|---|
| `image_assets` | `ImageTestAssets` |
| `video_assets` | `VideoTestAssets` |
| `audio_assets` | `AudioTestAssets` |
| `local_asset_server` | 基于线程的 HTTP 服务器，通过 URL 提供测试资源 |

### 生命周期管理（autouse）

| Fixture | 作用 |
|---|---|
| `cleanup_fixture` | **每个测试后**自动调用 `cleanup_dist_env_and_memory()` |
| `dynamo_reset` | 每个测试后重置 `torch._dynamo` 状态 |
| `init_test_http_connection` | 确保每个测试有独立 HTTP 客户端 |

子目录可以用 `@pytest.mark.skip_global_cleanup` 跳过全局清理（约快 10 倍）。

### 其他常用 Fixture

| Fixture | 用途 |
|---|---|
| `example_prompts` | 短文本提示（`tests/prompts/example.txt`） |
| `example_long_prompts` | 长文本提示（`tests/prompts/summary.txt`） |
| `sample_json_schema` | 结构化输出测试用 JSON Schema |
| `enable_pickle` | 设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` |
| `fake_vllm_ir` | 隔离的 IR 操作注册，每个测试独立命名空间 |
| `fresh_vllm_cache` | 临时缓存目录，避免测试污染 |
| `default_vllm_config` | 临时 `VllmConfig()`，用于不需要全引擎的算子测试 |
| `workspace_init` | 初始化 V1 工作区管理器 |

## 模型测试模式

### 模型注册表（`tests/models/registry.py`）

所有模型通过数据类 `_HfExamplesInfo` 注册，包含元数据：模型 ID、dtype、tokenizer、speculative_model、是否在线可用等。

架构到条目的映射在 `HF_EXAMPLE_MODELS` 字典中（key 为 HF `config.architectures[0]`）。

### 标准测试模板

```python
@pytest.mark.parametrize("model", [
    pytest.param("facebook/opt-125m", marks=[pytest.mark.core_model, pytest.mark.cpu_model]),
    pytest.param("meta-llama/Llama-3.2-1B", marks=[pytest.mark.core_model, large_gpu_mark(min_gb=48)]),
])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, example_prompts, model, max_tokens, num_logprobs):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate(example_prompts, max_tokens)
    with vllm_runner(model) as vllm_model:
        vllm_outputs = vllm_model.generate(example_prompts, max_tokens)

    check_logprobs_close(outputs_0_lst=hf_outputs, outputs_1_lst=vllm_outputs, ...)
```

### 验证工具（`tests/models/utils.py`）

| 函数 | 用途 |
|---|---|
| `check_outputs_equal` | 严格 token/text 精确匹配（贪婪解码） |
| `check_logprobs_close` | 宽松比较，检查 vLLM 的 top-N logprobs 与 HF 的 top-N 是否重叠 |

## Pytest 标记

| 标记 | 含义 |
|---|---|
| **`core_model`** | 高优先级模型测试，每个 PR 必跑 |
| **`cpu_model`** | 同时跑 CPU CI 的模型 |
| **`slow_test`** | 仅在夜间/慢速 CI 作业中运行 |
| **`distributed`** | 仅多 GPU 分布式测试中运行 |
| **`optional`** | 默认跳过，需 `--optional` 标志才会运行 |
| **`skip_global_cleanup`** | 跳过 autouse 清理（不碰 torch/cuda 的测试） |
| **`split`** | 可在拆分测试会话中运行（CI 负载均衡） |
| **`hybrid_model`** | 含 Mamba/SSM + Attention 混合层的模型 |

### 条件跳过装饰器

| 装饰器 | 条件 |
|---|---|
| `large_gpu_mark(min_gb=N)` | GPU 显存不足 N GB 时跳过 |
| `multi_gpu_test(num_gpus=N)` | 可用 GPU 不足 N 个时跳过 |
| `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` | 无 CUDA 时跳过 |

## 内核测试特点

- 直接调用内核函数，**不启动完整 LLM 引擎**
- 大量使用 `@pytest.mark.parametrize` 扫描参数空间（dtype、seq_len、num_heads、head_size）
- 使用 `tests/kernels/utils.py` 中的 `QKVInputs`、`QKVO`、`make_tensor_with_pad` 构建测试张量
- 使用 `tests/kernels/allclose_default.py` 定义容差阈值
- `tests/kernels/attention/` — 最大子集（35+ 文件）：FlashAttention、FlashInfer、Triton、MLA、ROCm
- `tests/kernels/moe/` — MoE 内核：CUTLASS、DeepGEMM、DeepEP、FlashInfer MoE、top-k 路由
- `tests/kernels/core/` — 核心操作：激活函数、RMS Norm、RoPE、FP8

## 多 GPU 分布式测试

- 位于 `tests/distributed/`，有自己的 `conftest.py`（端口分配、ZMQ 发布/订阅对）
- 覆盖：TP/PP/EP/CP 并行、custom all-reduce、pipeline parallel、expert placement、Ray V2 执行器
- 使用 `multi_gpu_test(num_gpus=N)` 装饰器跳过 GPU 不足的场景

## 编译测试（`tests/compile/`）

三个层次：
1. 根级 `test_*.py` — 单元测试（自动被 CI 捡起）
2. `fullgraph/` — 完整模型编译测试（自动被 CI 捡起）
3. `correctness_e2e/`、`fusions_e2e/` — 需要多 GPU 的端到端测试
4. `passes/` — 编译器 pass 测试（融合、功能化、lowering）

`compile/conftest.py` 提供 `mock_cuda_platform` 和 `mock_xpu_platform` fixture。

## `--optional` 可选测试机制

根 conftest 中的 `pytest_addoption` 添加 `--optional` 标志。所有 `@pytest.mark.optional` 标记的测试默认跳过，仅当 `--optional` 传递时才运行。用于轻量级检查 vs 完整测试套件。
