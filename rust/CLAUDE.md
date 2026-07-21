# Rust 前端 CLAUDE.md

本文件对 `rust/` 目录提供指导。阅读时请结合项目级 [CLAUDE.md](../CLAUDE.md) 及 [AGENTS.md](AGENTS.md)。

## Workspace 结构

14 个 crate，严格分层，每个只依赖其下方一层：

```
vllm-cmd (+ mimalloc)              ← 二进制入口
  ├── vllm-server                  ← HTTP/gRPC API 服务
  │     ├── vllm-chat              ← 聊天模板、推理/工具调用解析
  │     │     ├── vllm-text        ← 文本级 facade（编码+增量解码）
  │     │     │     ├── vllm-llm   ← generate/abort facade
  │     │     │     │     ├── vllm-engine-core-client  ← ZMQ 通信
  │     │     │     │     │     ├── vllm-metrics       ← Prometheus 指标
  │     │     │     │     │     └── zeromq
  │     │     │     │     └── vllm-metrics
  │     │     │     └── vllm-tokenizer
  │     │     └── vllm-parser      ← 工具调用/推理解析器（winnow）
  │     │           └── vllm-tokenizer
  │     └── vllm-llm
  ├── vllm-bench                   ← 独立，无内部依赖
  ├── vllm-managed-engine          ← 独立，管理 Python 引擎子进程
  └── vllm-mock-engine             ← 用于压力测试的 mock 引擎
        └── vllm-engine-core-client
```

## 各 Crate 职责

### engine-core-client — Python 引擎通信（底层）

通过 **ZeroMQ ROUTER/DEALER socket + msgpack** 与 Python `EngineCoreProc` 通信。

核心类型：
- `EngineCoreClient` — 持有 ZMQ runtime、输入/输出 socket、后台任务
- `EngineCoreClientConfig` — 两种传输模式：`HandshakeOwner`（Rust 控制握手）、`Bootstrapped`（Python 提供固定地址）
- `EngineId` — 两字节 ZMQ 路由标识

关键方法：`call(req) → EngineCoreOutputStream`、`abort(ids)`、`call_utility(method, args)`、`collective_rpc(...)`

### llm — LLM Facade

封装 `EngineCoreClient`，提供 `generate()` / `abort()`。管理请求 ID 映射（外部→内部）、指标追踪、统计日志。

### text — 文本 Facade

在 llm 之上增加 prompt 编码和增量解码：
- `TextLlm.generate(request)` → 解码后的文本事件流
- `TextLlm.generate_raw(request)` → 原始 token 流
- `TextBackend::HfTextBackend` — HuggingFace tokenizer 后端

### chat — 聊天 Facade

在 text 之上增加聊天模板、推理解析、工具调用解析：
- `ChatLlm.chat(request)` → `ChatEventStream`（`TextDelta | ReasoningDelta | ToolCallDelta | Finished`）
- `ChatRenderer` trait（minijinja 模板、Harmony 协议、DeepSeek 特定）
- `ChatOutputProcessor` — 增量解析工具调用和推理内容

### server — HTTP/gRPC API 服务

提供 OpenAI 兼容 API：

| 路由 | 功能 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /metrics` | Prometheus 指标 |
| `GET /v1/models` | 模型列表 |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/chat/completions` | 聊天补全 |
| `POST /tokenize` | 分词 |
| `POST /detokenize` | 解分 |
| `POST /v1/load_lora_adapter` | 加载 LoRA（可选） |

中间件：请求 ID、API Key 鉴权、CORS、服务端负载追踪、HTTP 指标。

gRPC：`vllm_grpc.proto` 定义 `Generate` 和 `GenerateStream` RPC。

### managed-engine — 引擎子进程管理

- `ManagedEngineConfig` — 配置 `python -m vllm.entrypoints.cli.main serve <model> --headless ...`
- `ManagedEngineHandle` — RAII 包装，`spawn()` / `shutdown(timeout)`（SIGTERM → 超时 → SIGKILL）
- `allocate_handshake_port(host)` — 临时端口分配
- `repartition_managed_engine_args()` — 复杂的参数分割：Rust 识别的参数留在前，未识别的转发给 Python

### parser — 工具调用/推理解析器

三个子模块：

**tool/** — 22 个注册的工具调用解析器，实现 `ToolParser` trait。命名：`deepseek_v3`、`llama3_json`、`qwen3_xml`、`glm45` 等。

**reasoning/** — 17 个推理解析器，实现 `ReasoningParser` trait。设计要点：解析器初始化时检查 **prompt token IDs** 推断最后一个推理边界（而非硬编码模型族约定）。

**python/** — PyO3 绑定，将 `vllm-parser` 暴露给 Python 作为 `vllm._rust_tool_parser`。

### bench — 基准测试

高性能 `vllm bench serve` 的 Rust 重写。独立，无内部 crate 依赖。

### mock-engine — Mock 引擎

用于前端压力测试的无头 mock 引擎。依赖 `engine-core-client`。

## 编码约定（摘要，详见 AGENTS.md）

- **Workspace 依赖**：始终通过 workspace Cargo.toml 引用依赖
- **小模块**：拆分为小文件，避免巨型单文件
- **Winnow 解析器**：声明式优先，元组组合优先于 `parse_next`，内置组合子优先于自定义。每个解析函数加 `Parse a ...` 注释
- **错误处理**：
  - 禁止对错误值直接调用 `to_string()`，使用 `ToReportString` / `AsReport`
  - 自由文本错误用 struct variant + `message: String` + `thiserror_ext::Macro`
  - 表达式位置用 `foo!(...)`，提前返回用 `bail_foo!(...)`
- **API 稳定性**：早期阶段可打破 API
- **平台**：仅 Unix，无需 `cfg(unix)`
- **测试**：快照测试（`expect_test`），`for_test()` fixture + struct update 语法，确定性同步（channel/barrier）优于 `sleep`，使用 `cargo nextest run`

## 常用命令

```bash
cd rust

# 编译
cargo build --release

# 全部测试（推荐 nextest）
cargo nextest run

# 单个 crate 测试
cargo nextest run -p vllm-parser

# 更新快照
UPDATE_EXPECT=1 cargo test

# 编译 proto
# proto 文件: proto/vllm_grpc.proto（通过 tonic-build 自动编译）
```
