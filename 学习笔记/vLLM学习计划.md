# vLLM 学习计划与面试题集

## 学习路线图

```
阶段一：基础概念（1周）
  → 阶段二：快速上手（1周）
    → 阶段三：核心原理（2周）
      → 阶段四：性能调优（2周）
        → 阶段五：生产部署（2周）
          → 阶段六：源码深入（3周）
```

---

## 阶段一：LLM 推理基础概念（1周）

### 学习目标
理解 LLM 推理的基本流程和核心概念，为后续学习 vLLM 打基础。

### 学习内容

| 序号 | 主题 | 要点 |
|---|---|---|
| 1.1 | Transformer 推理过程 | Prefill vs Decode 阶段、自回归生成、每个 step 的计算特征 |
| 1.2 | KV Cache 是什么 | 为什么需要 KV Cache、KV Cache 的内存占用计算（`2 × layers × hidden_dim × seq_len × dtype_bytes`） |
| 1.3 | 显存构成 | 模型权重 + KV Cache + 激活值 + 临时缓冲区，各自的占比估算 |
| 1.4 | 吞吐 vs 延迟 | Throughput（tokens/s）、TTFT（Time To First Token）、TPOT（Time Per Output Token）、latency SLA |
| 1.5 | 传统推理的瓶颈 | 静态 batching 的内存浪费、请求长度差异导致的 padding 开销 |
| 1.6 | LLM 服务 vs 传统 Web 服务 | 有状态、长连接、流式输出、GPU 显存敏感 |

### 动手实践
```bash
# 用 HuggingFace 跑一次原始推理，感受 KV Cache 和 decode 循环
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
# 观察 prefill 输入 → model.forward → logits → next token 的完整流程
"
```

---

## 阶段二：vLLM 快速上手（1周）

### 学习目标
能在本地启动 vLLM 服务、调用 API、运行 benchmark，建立基本使用能力。

### 学习内容

| 序号 | 主题 | 要点 |
|---|---|---|
| 2.1 | 安装部署 | pip install、Docker 部署、环境变量配置 |
| 2.2 | vLLM serve 启动 | CLI 参数详解：`--model`、`--tensor-parallel-size`、`--max-model-len`、`--gpu-memory-utilization` |
| 2.3 | OpenAI 兼容 API | `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/models` |
| 2.4 | 离线推理 | `LLM` 类的使用：`llm.generate()`、`llm.encode()`、`SamplingParams` |
| 2.5 | Benchmark | `vllm bench serve` 和 `vllm bench throughput`，理解 RPS、latency 指标 |

### 动手实践
```bash
# 1. 启动服务
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000

# 2. 调用 API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"你好"}]}'

# 3. 跑 benchmark
vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 --request-rate 10
```

---

## 阶段三：核心原理（2周）

### 学习目标
深入理解 vLLM 的核心创新：PagedAttention、Continuous Batching、KV Cache 管理。

### 学习内容

| 序号 | 主题 | 要点 |
|---|---|---|
| 3.1 | PagedAttention 原理 | 类比 OS 虚拟内存分页：将 KV Cache 从连续分配改为 block 级分配，消除内部/外部碎片 |
| 3.2 | Block Table | block table 的寻址方式、logical block → physical block 映射 |
| 3.3 | Continuous Batching | 与传统 static batching 的区别、如何在每个 step 动态加入/移除请求 |
| 3.4 | Chunked Prefill | 将长 prompt 的 prefill 分块处理，避免大 prefill 阻塞 decode |
| 3.5 | Prefix Caching | 相同前缀的请求共享 KV Cache block、cache_salt 与哈希匹配机制 |
| 3.6 | 调度策略 | FCFS vs Priority、抢占机制（preemption）、scheduler 决策流程 |
| 3.7 | V1 引擎架构 | Scheduler-Core-Worker 三层架构、ZMQ 通信、异步流水线 |

### 动手实践
- 阅读论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- 阅读源码：`vllm/v1/core/sched/scheduler.py` 的 `schedule()` 方法
- 实验：相同模型不同 `--max-num-seqs` 对吞吐的影响

---

## 阶段四：性能调优（2周）

### 学习目标
掌握 vLLM 各项性能优化技术，能针对不同场景做配置调优。

### 学习内容

| 序号 | 主题 | 要点 |
|---|---|---|
| 4.1 | 量化推理 | FP8、INT8、INT4、GPTQ、AWQ 的精度-速度权衡，`--quantization` 参数 |
| 4.2 | Tensor Parallelism | 张量并行的原理、`--tensor-parallel-size` 配置、通信开销分析 |
| 4.3 | Pipeline Parallelism | 流水线并行的微批次调度、bubble 率、PP 与 TP 的配合 |
| 4.4 | CUDA Graph | CUDA Graph 捕获与重放、减少 kernel launch overhead、`--enforce-eager` 对比 |
| 4.5 | FlashAttention | FlashAttention v2/v3 原理（tiling + recomputation）、vLLM 中的后端选择 |
| 4.6 | 投机解码 | Speculative Decoding：draft model + target model 的配合、接受率对吞吐的影响 |
| 4.7 | 显存优化 | `--gpu-memory-utilization`、`--max-model-len`、`cpu-offload-gb` 的调优策略 |
| 4.8 | 请求调度优化 | `--max-num-seqs`、`--max-num-batched-tokens` 的 trade-off |

### 动手实践
```bash
# 对比不同配置的吞吐
# 基准
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --request-rate inf

# 开启 FP8 量化
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --quantization fp8 --request-rate inf

# 开启 TP=2
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --request-rate inf

# 对比 TensorRT-LLM、SGLang 等竞品的性能差异曲线
```

---

## 阶段五：生产部署（2周）

### 学习目标
掌握生产环境的部署、监控、容错、多节点扩展。

### 学习内容

| 序号 | 主题 | 要点 |
|---|---|---|
| 5.1 | 多节点部署 | Ray 集群、`--pipeline-parallel-size`、数据并行（DP）、disaggregated prefill/decode |
| 5.2 | 监控与可观测 | Prometheus metrics（TTFT、TPOT、queue time、cache hit rate、GPU util）、OpenTelemetry tracing |
| 5.3 | 健康检查与容错 | `/health`、`/load` 端点、OOM recovery、engine restart 策略 |
| 5.4 | 负载均衡 | 多引擎实例的负载分发、request routing、session affinity |
| 5.5 | LoRA 热加载 | 动态加载/卸载 LoRA 适配器、`--enable-lora`、`--max-lora-rank` |
| 5.6 | 结构化输出 | JSON mode、grammar-guided generation、xgrammar 后端 |
| 5.7 | 安全与限流 | API Key 鉴权、rate limiting、并发控制 |
| 5.8 | 常见故障排查 | OOM 诊断、精度异常（NaN）、吞吐下降根因分析 |

### 动手实践
```bash
# 多实例部署
vllm serve model-A --port 8000 &
vllm serve model-B --port 8001 &

# 抓取 Prometheus 指标
curl http://localhost:8000/metrics | grep vllm

# 结构化输出
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-0.5B-Instruct",
    "messages":[{"role":"user","content":"返回一个JSON: name和age"}],
    "response_format":{"type":"json_object"}
  }'
```

---

## 阶段六：源码深入（3周）

### 学习目标
能阅读和修改 vLLM 源码、贡献 PR、支持自定义模型。

### 学习内容

| 序号 | 主题 | 要点 |
|---|---|---|
| 6.1 | 引擎主循环 | `EngineCore.step()` 的完整流程：schedule → execute → sample → update |
| 6.2 | Attention 后端 | `selector.py` 的选择逻辑、如何添加新的 attention backend |
| 6.3 | 模型注册 | `ModelRegistry`、自定义模型接入（HF config → vLLM model adapter） |
| 6.4 | 量化插件 | quantization config/method 的插件架构 |
| 6.5 | 分布式执行 | `multiproc_executor.py` 的 `collective_rpc` 机制 |
| 6.6 | 投机解码 | `rejection_sampler.py`、EAGLE/Medusa proposer |
| 6.7 | 编译优化 | `piecewise_backend.py`、fusion passes、CUDA Graph 捕获流程 |

### 动手实践
- 为一个未支持的 HuggingFace 模型添加 vLLM 支持
- 阅读 `gpu_model_runner.py`，理解输入准备和 forward 流程
- 调试一个推理请求从 API 到 GPU kernel 的完整路径

---

## 推荐学习资源

| 资源 | 类型 | 链接 |
|---|---|---|
| PagedAttention 论文 | 论文 | https://arxiv.org/abs/2309.06180 |
| vLLM 官方文档 | 文档 | https://docs.vllm.ai |
| vLLM Blog | 博客 | https://blog.vllm.ai |
| FlashAttention 论文 | 论文 | https://arxiv.org/abs/2205.14135 |
| Continuous Batching 博客 | 博客 | https://www.anyscale.com/blog/continuous-batching-llm-inference |
| AWQ 量化论文 | 论文 | https://arxiv.org/abs/2306.00978 |
| GPTQ 量化论文 | 论文 | https://arxiv.org/abs/2210.17323 |
| Speculative Decoding 论文 | 论文 | https://arxiv.org/abs/2211.17192 |
| vLLM 源码 | 代码 | https://github.com/vllm-project/vllm |
