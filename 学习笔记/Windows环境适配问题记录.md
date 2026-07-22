# Windows 环境适配问题记录

vLLM 官方仅支持 Linux，在 Windows 下开发/学习需要做以下适配。

## 问题一：uvloop 不可用

### 现象

导入 vLLM 模块时报错：
```
ModuleNotFoundError: No module named 'uvloop'
```

### 原因

`uvloop` 是基于 libuv 的 asyncio 事件循环替代方案，使用 Unix 专有 API（`epoll`/`kqueue`），Windows 不支持。vLLM 在以下 7 个文件中硬编码了 `import uvloop`：

| 文件 | 用途 |
|---|---|
| `vllm/v1/utils.py` | API 服务启动 `uvloop.run()` |
| `vllm/entrypoints/openai/api_server.py` | OpenAI API 服务器 |
| `vllm/entrypoints/openai/dp_supervisor.py` | Data Parallel 管理 |
| `vllm/entrypoints/cli/serve.py` | CLI `vllm serve` 命令 |
| `vllm/entrypoints/cli/launch.py` | CLI 启动器 |
| `vllm/entrypoints/grpc_server.py` | gRPC 服务器 |
| `vllm/benchmarks/throughput.py` | 吞吐量 benchmark |

### 修复

将 `import uvloop` 改为可选导入，不可用时降级为标准 `asyncio.run()`：

```python
# 修改前
import uvloop

# 修改后
try:
    import uvloop
except ImportError:
    uvloop = None  # type: ignore[assignment]
```

并将所有 `uvloop.run(xxx)` 调用替换为 `_run_uvloop_async(xxx)`，辅助函数：

```python
def _run_uvloop_async(coro):
    """uvloop 不可用时降级为 asyncio.run()"""
    if uvloop is not None:
        uvloop.run(coro)
    else:
        import asyncio
        asyncio.run(coro)
```

---

## 问题二：无 GPU 时 `torch.accelerator.empty_cache()` 报错

### 现象

测试通过但 cleanup fixture 报错：
```
RuntimeError: Cannot access accelerator device when none is available.
```

### 原因

`vllm/distributed/parallel_state.py` 的 `cleanup_dist_env_and_memory()` 中调用 `torch.accelerator.empty_cache()` 时，如果系统没有 GPU 或加速器，PyTorch 的 accelerator 未初始化，导致报错。

### 修复

在 `parallel_state.py` 中给 `empty_cache()` 加异常保护：

```python
# 修改前
if not current_platform.is_cpu():
    torch.accelerator.empty_cache()

# 修改后
if not current_platform.is_cpu():
    try:
        torch.accelerator.empty_cache()
    except RuntimeError:
        pass
```

---

## 修改文件清单

| 文件 | 修改内容 |
|---|---|
| `vllm/v1/utils.py` | uvloop 可选导入 + asyncio 降级 |
| `vllm/entrypoints/openai/api_server.py` | uvloop 可选导入 + `_run_uvloop_async` 函数 |
| `vllm/entrypoints/openai/dp_supervisor.py` | uvloop 可选导入 + `_run_uvloop_async` 函数 |
| `vllm/entrypoints/cli/serve.py` | uvloop 可选导入 + `_run_uvloop_async` 函数 |
| `vllm/entrypoints/cli/launch.py` | uvloop 可选导入 + `_run_uvloop_async` 函数 |
| `vllm/entrypoints/grpc_server.py` | uvloop 可选导入 + `_run_uvloop_async` 函数 |
| `vllm/benchmarks/throughput.py` | uvloop 可选导入 + `_run_uvloop_async` 函数 |
| `vllm/distributed/parallel_state.py` | `empty_cache()` 异常保护 |

## 注意事项

- 这些修改仅用于 Windows 开发/学习环境，不影响 Linux 上的正常行为
- vLLM 的 **模型推理功能**（GPU kernel、CUDA graph 等）在 Windows 上仍不可用，仅能跑 CPU 相关的单元测试和代码研究
- 学习源码和运行测试足够了，生产部署请用 Linux
