# vLLM 面试题集

题目按难度分级：⭐ 基础 / ⭐⭐ 进阶 / ⭐⭐⭐ 深入

---

## 一、LLM 推理基础

### Q1-1：Transformer 推理分为哪两个阶段？各有什么特点？⭐⭐

**答：** Prefill 阶段和 Decode 阶段。

| 维度 | Prefill | Decode |
|---|---|---|
| 输入 | 全部 prompt tokens | 仅上一个生成的 token |
| 计算特征 | 计算密集，矩阵乘法大 | 访存密集，受限于显存带宽 |
| KV Cache | 写入新 KV | 追加 1 token 的 KV |
| 延迟敏感度 | 影响 TTFT | 影响 TPOT |
| 并行度 | 高，可利用 tensor 并行 | 低，单 token 计算量小 |

### Q1-2：KV Cache 的内存占用如何计算？一个 70B 模型，seq_len=4096 时 KV Cache 占多少显存？⭐⭐

**答：** 公式：

```
KV Cache = 2 × num_layers × hidden_dim × seq_len × dtype_bytes × batch_size
```

以 Llama-70B 为例（num_layers=80, hidden_dim=8192, FP16 dtype=2B）：

```
单请求：2 × 80 × 8192 × 4096 × 2 ≈ 10 GB
batch_size=8：≈ 80 GB
```

这解释了为什么推理服务中 KV Cache 是显存瓶颈。

### Q1-3：什么是 TTFT 和 TPOT？为什么两者需要权衡？⭐

**答：**
- **TTFT（Time To First Token）**：从请求到达到第一个 token 生成的时间，用户体验的关键
- **TPOT（Time Per Output Token）**：每个后续 token 的生成时间（不含第一个），影响长文本生成的总延迟

权衡关系：增大 batch size 提高吞吐，但每个请求的 TTFT 和 TPOT 会因排队和计算竞争而上升。

---

## 二、PagedAttention 原理

### Q2-1：PagedAttention 解决了什么问题？核心思想是什么？⭐⭐

**答：** 解决了传统 KV Cache 管理的内存浪费问题。

**传统方式的问题：**
- 为每个请求预分配 `max_seq_len` 长度的连续 KV Cache → 大量内部碎片
- 不同请求的 KV Cache 分配在不同位置 → 无法共享

**PagedAttention 解决方案：**
- 将 KV Cache 切分为固定大小的 block（如 16 tokens/block）
- 通过 Block Table 将逻辑位置映射到物理 block
- 类比 OS 虚拟内存：逻辑地址连续，物理存储可以不连续
- 实现：请求间可共享 block（prefix caching）、按需分配（消除预留浪费）

### Q2-2：PagedAttention 的 block_size 如何选择？⭐⭐

**答：**

| block_size | 优点 | 缺点 |
|---|---|---|
| 小（8-16） | 碎片少、共享粒度高 | Block Table 更大、寻址开销增加 |
| 大（32-64） | 寻址开销小、内存更连续 | 碎片多、共享效率低 |

vLLM 默认 `block_size=16`，是内存效率和计算开销的平衡点。大模型（70B+）可考虑增大以降低 Block Table 开销。

### Q2-3：请求被抢占（preemption）时，PagedAttention 如何处理 KV Cache？⭐⭐⭐

**答：** 抢占策略有两种：

1. **Recompute（重计算）**：丢弃被抢占请求的 KV Cache，恢复时重新 prefill → 简单但增延迟
2. **Swap（交换）**：将 KV Cache blocks 换出到 CPU 内存，恢复时换回 GPU → 需要 CPU-GPU 带宽

vLLM 默认使用 swap 策略（优先），当 CPU swap 空间不足时回退到 recompute。核心是 block 级别的灵活管理——不需要整体换出，只换出被抢占请求占用的 block。

---

## 三、Continuous Batching

### Q3-1：Static Batching 和 Continuous Batching 的区别？⭐⭐

**答：**

| 维度 | Static Batching | Continuous Batching |
|---|---|---|
| 批次构成 | 固定，batch 中所有请求必须同步完成 | 动态，每步可加入/移除请求 |
| 短请求处理 | 等待整个 batch 完成，延迟高 | 完成后立即退出 batch，延迟低 |
| GPU 利用率 | 短长混合时利用率低 | 始终保持高利用率 |
| 实现复杂度 | 简单 | 复杂，需 KV Cache 动态管理 |

**举例：** 一个 batch 中有 2 个请求，分别生成 10 个和 100 个 token。Static batching 下短请求需等待 90 步（大部分计算浪费），Continuous batching 下短请求完成即退出，新请求立即加入。

### Q3-2：vLLM 调度器如何决定每一步调度哪些请求？⭐⭐

**答：** Scheduler 每步的决策流程：

1. **排序 running 请求**：按 FCFS 或 priority 排序已在运行的请求
2. **分配 KV Cache blocks**：为每个请求的这一步分配 block（续写或新分配）
3. **从 waiting 队列取新请求**：当前步有剩余 token budget 时取
4. **输出 SchedulerOutput**：`scheduled_new_reqs`、`scheduled_running_reqs`、`num_scheduled_tokens`

关键约束：
- `max_num_seqs`：最大并发请求数
- `max_num_batched_tokens`：每步最大调度 token 数
- GPU 显存中可用 block 数

### Q3-3：什么是 Chunked Prefill？为什么需要它？⭐⭐⭐

**答：** 将长 prompt 的 prefill 分块处理，每次只处理 token_budget 允许的 chunk。

**问题场景：** 一个 32K tokens 的超长 prompt 到达时，如果一次性 prefill：
- 计算量巨大，阻塞所有 decode 请求数秒
- 大量 KV Cache block 分配可能挤占其他请求

**Chunked Prefill 的做法：**
- 将 prefill 拆为多个 chunk（如每次 2048 tokens）
- 每步执行：一个 prefill chunk + 若干 decode
- TTFT 变为"首 chunk prefill 时间"而非"全 prompt prefill 时间"

---

## 四、Prefix Caching

### Q4-1：Prefix Caching 的匹配机制是什么？⭐⭐

**答：**

1. 每个 block 的内容计算一个 hash（基于其中的 token sequence）
2. 调度时，计算新请求已计算的 token 序列的 hash
3. 在全局 Block Table 中查找相同 hash 的 block
4. 命中则直接复用 block，跳过 prefill 计算

**匹配粒度：** block 级别。只有完整的 16 token（一个 block）才能被共享。

### Q4-2：什么场景下 Prefix Caching 效果最好？⭐⭐

**答：**

- **System prompt 固定**：每个请求共享同一段系统提示词 → 100%命中率
- **Few-shot 示例相同**：多个请求共享 few-shot examples → 示例部分命中
- **RAG 场景**：多个请求使用相同的检索上下文 → context 部分命中
- **多轮对话**：后续轮次与前一轮共享历史 → 历史部分增量命中

典型收益：system prompt 2000 tokens + 高并发 → 节省 30-50% prefill 计算。

---

## 五、分布式推理

### Q5-1：Tensor Parallelism（TP）的原理和通信开销？⭐⭐

**答：**

**原理：** 将每层的权重矩阵按列切分到多个 GPU，每张 GPU 计算一部分，最后通过 All-Reduce 汇总。

**通信开销：**
- 每层两次 All-Reduce（attention output + FFN output）
- 通信量与 `batch_size × hidden_dim` 成正比

**适用场景：**
- 单卡放不下模型权重 → TP 是刚需
- 单卡放得下但 prefill 太慢 → TP 加速矩阵乘法
- 小 batch 的 decode 阶段 TP 收益有限（通信开销 > 计算收益）

### Q5-2：Pipeline Parallelism（PP）与 TP 的适用场景区别？⭐⭐⭐

**答：**

| 维度 | TP | PP |
|---|---|---|
| 通信量 | 大（每层 All-Reduce） | 小（仅层间激活值传输） |
| 通信频率 | 每层都通信 | 每个 micro-batch 通信一次 |
| 显存节省 | 每张卡存全部层的一部分 | 每张卡存部分层 |
| GPU 利用率 | 高（所有 GPU 同时计算） | 有 pipeline bubble |
| 跨节点 | 不适合（NCCL 跨节点慢） | 适合（通信量小） |

**黄金组合：** 节点内 TP（利用 NVLink 高带宽）+ 节点间 PP（利用小通信量跨节点）。

### Q5-3：vLLM 如何实现多节点分布式推理？⭐⭐

**答：** 通过 Ray 框架管理多节点资源。步骤：

1. 启动 Ray 集群（`ray start --head` + `ray start --address=...`）
2. `VLLM_USE_RAY_COMPILED_DAG=1` 启用 compiled DAG 传输
3. 配置 `--tensor-parallel-size` 和 `--pipeline-parallel-size`
4. vLLM 通过 Ray placement group 分配 GPU，通过 `ray_executor.py` 协调

V1 引擎支持通过 `ray_executor_v2.py` 使用 compiled DAG 优化跨节点通信。

---

## 六、量化

### Q6-1：FP8、INT8、INT4 量化的精度-速度权衡？⭐⭐

**答：**

| 量化格式 | 模型大小（相对 FP16） | 推理速度 | 精度损失 | 适用模型 |
|---|---|---|---|---|
| FP8 | 50% | 1.5-2x | 极小 | 原生支持 FP8 的 GPU（H100+） |
| INT8 (AWQ/GPTQ) | 50% | 1.3-1.8x | 小 | 所有 CUDA GPU |
| INT4 (AWQ/GPTQ) | 25% | 1.5-2.5x | 中等 | 显存紧缺场景 |
| W4A16 (AWQ) | 25% | 加速有限 | 可接受 | 仅量化权重，激活保持 FP16 |

**关键认知：** 量化主要节省显存，速度提升取决于硬件对量化计算的原生支持（如 H100 的 FP8 tensor core）。

### Q6-2：AWQ 和 GPTQ 的主要区别？⭐⭐

**答：**

- **GPTQ**：逐层优化，使用二阶信息（Hessian）做最优量化，需要校准数据，量化过程较慢
- **AWQ**：观察权重通道的重要性（通过激活值的幅值），只保护 1% 的关键通道，量化过程更快

两者在精度上相当，AWQ 的工程实现更简单。

### Q6-3：vLLM 的量化插件架构是怎样的？⭐⭐⭐

**答：** vLLM 中量化采用 plugin 架构，每种方案提供三个组件：

1. **QuantizationConfig**：量化参数配置（如 `bits`、`group_size`、`sym`）
2. **QuantLinearMethod**：量化后的 Linear 层实现（如 `awq::AWQLinearMethod`）
3. **FusedMoEMethod**（可选）：量化后的 MoE 层实现

在 `vllm/model_executor/layers/quantization/` 下，每种方案一个子目录，通过 `ModelConfig.quantization` 参数选择。

---

## 七、性能优化实战

### Q7-1：一个请求的延迟突然飙高，如何排查？⭐⭐⭐

**答：** 排查路径：

1. **队列等待** → 检查 `vllm:request_queue_time_seconds` 指标
2. **Prefill 慢** → 检查 prompt 长度、是否命中 prefix cache
3. **Decode 慢** → 检查并发请求数、是否触发抢占
4. **GPU 显存** → `nvidia-smi` 确认 OOM 或 swap 发生
5. **调度延迟** → `VLLM_LOGGING_LEVEL=DEBUG` 查看 scheduler 耗时
6. **模型编译** → 确认 CUDA Graph 已捕获（非 `--enforce-eager`）

### Q7-2：如何为一个新场景调优 vLLM 配置？⭐⭐

**答：** 调优流程：

1. **明确目标**：高吞吐 or 低延迟？长 prompt or 短 prompt？
2. **显存测算**：`GPU 显存 = 模型权重 + KV Cache + 激活值`，先确保放得下
3. **`gpu-memory-utilization`**：默认 0.9，OOM 时降到 0.85
4. **`max-num-seqs`**：越大吞吐越高但延迟越高，逐步调大观察 latency 拐点
5. **`max-model-len`**：设太大浪费 KV cache，设太小截断长请求
6. **量化**：显存不够先开 FP8，还不够上 INT4
7. **TP**：单卡放不下 or prefill 需求高时开
8. **benchmark 迭代**：`vllm bench serve` 跑不同 RPS，观察 P50/P99 latency

### Q7-3：CUDA Graph 如何加速推理？什么情况不适合？⭐⭐

**答：**

**加速原理：** 将 GPU kernel 调用序列一次性录制为"图"，后续直接重放，消除单次 kernel launch 的 CPU overhead（每次 launch 约 5-10μs，decode 阶段一个 batch 可能有数千次 launch）。

**不适合的场景：**
- 请求数波动大（batch size 变化 → graph 需重新捕获）
- 动态形状频繁变化
- 超长上下文（graph 捕获的显存开销大）

**vLLM 的做法：** 预捕获多种 batch size 的 graph，运行时按需选择。

---

## 八、生产部署

### Q8-1：生产环境中 vLLM 的关键监控指标有哪些？⭐⭐

**答：**

| 指标 | 含义 | 告警阈值 |
|---|---|---|
| `vllm:request_success_total` | 成功请求数 | 增长停止 → 服务异常 |
| `vllm:time_to_first_token_seconds` | TTFT P50/P99 | P99 > 500ms → prefill 瓶颈 |
| `vllm:time_per_output_token_seconds` | TPOT | 上升 → decode 瓶颈 |
| `vllm:request_queue_time_seconds` | 排队时间 | > 1s → 并发过大 |
| `vllm:num_requests_running` | 当前运行请求数 | 接近 max-num-seqs → 需扩容 |
| `vllm:num_requests_waiting` | 等待队列长度 | 持续 > 0 → 需扩容 |
| `vllm:gpu_cache_usage_perc` | KV Cache 使用率 | > 90% → 可能触发抢占 |
| `vllm:prefix_cache_hit_rate` | 前缀缓存命中率 | < 50% → 检查 cache 配置 |
| `vllm:request_prompt_tokens` | 输入 token 分布 | 突变 → 上游异常 |
| `vllm:request_generation_tokens` | 输出 token 分布 | 突变 → 应用逻辑变化 |

### Q8-2：如何处理推理服务的 GPU OOM？⭐⭐

**答：**

**预防：**
1. 监控 `gpu_cache_usage_perc`，设置告警
2. 预留 10-15% 显存作为 buffer
3. 设置 `--max-model-len` 限制单请求最大长度
4. 设置 `--max-num-seqs` 限制最大并发

**发生时：**
1. 立即降低 `--max-num-seqs` or `--max-model-len`
2. 启用 `--cpu-offload-gb` 将 KV Cache swap 到 CPU
3. 启用量化降低模型权重占用
4. 开 TP 分散权重到多卡
5. 实现 graceful degradation：超长请求返回 413 错误而非 crash

### Q8-3：多模型共存的部署架构？⭐⭐⭐

**答：** 三种模式：

1. **多实例单卡**（小模型）：一张 GPU 跑多个 vLLM 实例，各自绑定不同显存比例（`gpu-memory-utilization` 分配合计 < 1.0）
2. **一卡一模型**（中模型）：每张 GPU 一个实例，通过前端路由分发（基于 model name）
3. **多节点**（大模型）：Ray 集群，多个 model group 并存

路由方式：
- Nginx/Envoy 反向代理，基于 URL path 或 header 中的 model name 路由
- 自定义 gateway 做模型选择

---

## 九、与其他框架对比

### Q9-1：vLLM vs TensorRT-LLM 的优劣势？⭐⭐

| 维度 | vLLM | TensorRT-LLM |
|---|---|---|
| 易用性 | 高，pip install 即用 | 低，需编译引擎 |
| 性能 | 优秀 | 极致（手工优化 kernel） |
| 模型支持 | 200+，HF 即插即用 | 需手动适配 |
| 量化 | 支持多种，配置简单 | FP8/INT8 深度优化 |
| 开源社区 | 活跃，2000+ 贡献者 | NVIDIA 主导 |
| 适用场景 | 快速部署，模型多样性 | 固定模型，极致性能需求 |

### Q9-2：vLLM vs SGLang 的关键差异？⭐⭐

| 维度 | vLLM | SGLang |
|---|---|---|
| 编程模型 | OpenAI API 兼容 | RadixAttention + SGLang DSL |
| 结构化输出 | xgrammar/guidance/outlines | 原生 SGLang 语法 |
| RadixAttention | 不支持 | 支持（trie-based prefix sharing） |
| Disaggregated | Prefill/Decode 分离 | Prefill/Decode 分离 |
| 社区规模 | 更大 | 快速增长 |

---

## 十、CPU Offload 与 KV Cache 管理

### Q10-1：vLLM 的 CPU Offload 机制如何工作？什么场景适用？⭐⭐

**答：** `--cpu-offload-gb` 指定用于 swap KV Cache 的 CPU 内存量。

**触发条件：** GPU 显存不足以分配新请求的 KV Cache blocks 时。

**流程：**
1. Scheduler 选择被抢占的请求（优先级最低 or 最久未调度）
2. 将其 KV Cache blocks 从 GPU 拷贝到 CPU 内存
3. 释放 GPU blocks 供新请求使用
4. 请求恢复执行时，将 KV Cache blocks 拷贝回 GPU

**适用场景：** 突发流量时的缓冲区，长尾请求占着 block 不释放的场景。

### Q10-2：为什么 KV Cache 是推理服务的核心瓶颈？有哪些优化方向？⭐⭐⭐

**答：**

**为什么是瓶颈：**
- LLM 推理中 KV Cache 常占 30-60% 的 GPU 显存
- 每多一个并发请求，KV Cache 线性增长
- Release 后的碎片化影响利用率

**优化方向：**

1. **Prefix Caching**：共享相同前缀的 KV blocks
2. **KV Cache 量化**：KV Cache 本身也可以用 FP8/INT8 存储（如 `--kv-cache-dtype fp8`）
3. **Multi-Query / Grouped-Query Attention**：MQA 的 KV head=1，KV Cache 缩小数倍
4. **Multi-Head Latent Attention（MLA）**：将 KV 压缩为低秩潜在向量（DeepSeek-V2/V3 的核心创新），KV Cache 仅为原始大小的 1/5-1/10
5. **Layer-wise KV Cache**：跨层共享 KV Cache
6. **CPU/SSD Swap**：分层存储，热数据在 GPU，冷数据在 CPU/SSD

---

## 面试模拟：综合应用题

### Q11：从零部署一个 70B 模型给 100 个用户并发使用，你会怎么做？

**参考回答框架：**

1. **硬件选型**：70B FP16 ≈ 140GB + KV Cache。至少 4×A100-80G 或 2×H100-80G（用 FP8）
2. **启动配置**：`--tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.9 --quantization fp8`
3. **容量估算**：KV Cache per request ≈ 2×80×8192×4096×2 = 10GB（FP16），100 并发需 1TB 不可能 → 需要排队机制 + 限制 max-num-seqs=32
4. **延迟 SLA**：TTFT < 2s，TPOT < 50ms → 跑 benchmark 验证
5. **监控**：Prometheus + Grafana，TTFT P99、TPOT、cache hit rate、GPU util
6. **容错**：多实例 + health check + 自动重启
7. **优化迭代**：prefix caching for system prompt、量化调优、speculative decoding

### Q12：vLLM 的 V1 引擎相比 V0 做了什么改进？

**答：**

1. **架构统一**：V0 的 prefill/decode 分离模型变为 V1 的统一 `num_computed_tokens` 追踪
2. **异步流水线**：`AsyncScheduler` 使调度和执行可重叠，消除 pipeline bubble
3. **插件化 Attention**：33+ 种 attention 后端通过 `AttentionBackendEnum` 注册，按模型/硬件自动选择
4. **更简洁的 Executor 抽象**：`collective_rpc` 统一了所有 worker 通信
5. **Micro-batching**：PP 场景下自动拆分微批次

---

## 快速自测清单

完成学习后，你应该能：

- [ ] 解释 PagedAttention 的 block 管理机制
- [ ] 画出 Continuous Batching 的请求生命周期
- [ ] 计算任意模型 + 配置的显存占用
- [ ] 调优 `max-num-seqs`、`gpu-memory-utilization`、`max-model-len` 三个参数
- [ ] 解释 TP/PP 的原理并通过 benchmark 验证收益
- [ ] 说出 FP8/INT8/INT4 量化的精度-速度权衡
- [ ] 列出生产环境 5 个以上关键监控指标
- [ ] 追踪一个请求从 API → EngineCore → GPU Kernel 的路径
- [ ] 对比 vLLM vs TensorRT-LLM vs SGLang 的选型逻辑
- [ ] 说出现有 Prefix Caching 机制的局限性和改进方向
