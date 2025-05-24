# Gemma 模型集成

这个模块提供了两种使用 Google Gemma 模型的实现方式：

1. **MLX 版本** (`llm_gemma_mlx.py`) - 使用 MLX 框架，针对 Apple Silicon 优化
2. **Transformers 版本** (`llm_gemma_transfomers.py`) - 使用 Hugging Face Transformers，支持 4bit 量化

## 目录结构

```
llm/
├── base_gemma.py              # 公共基类，包含共享功能
├── llm_gemma_mlx.py          # MLX 版本实现
├── llm_gemma_transfomers.py  # Transformers 版本实现
├── example_usage.py          # 使用示例
└── README.md                 # 本文档
```

## 安装依赖

### MLX 版本
```bash
pip install mlx-lm
```

### Transformers 版本
```bash
pip install transformers torch
# 如果需要 4bit 量化支持
pip install bitsandbytes
```

## 基本使用

### MLX 版本

```python
from llm_gemma_mlx import GemmaMLXChatCompletion

# 初始化（默认使用 4bit 量化模型）
client = GemmaMLXChatCompletion()

# 或指定其他模型
client = GemmaMLXChatCompletion(
    model_name="mlx-community/gemma-3-12b-it-4bit-DWQ"
)

# 发送消息
messages = [
    {"role": "user", "content": "你好，介绍一下你自己。"}
]

response = client.create(
    messages=messages,
    temperature=0.7,
    max_tokens=100
)

print(response["choices"][0]["message"]["content"])
```

### Transformers 版本

```python
from llm_gemma_transfomers import GemmaTransformersChatCompletion

# 初始化（无量化）
client = GemmaTransformersChatCompletion()

# 或启用 4bit 量化
client = GemmaTransformersChatCompletion(
    model_name="google/gemma-3-12b-it",
    use_4bit_quantization=True
)

# 发送消息
messages = [
    {"role": "user", "content": "写一个 Python 函数计算斐波那契数列。"}
]

response = client.create(
    messages=messages,
    temperature=0.7,
    max_tokens=200
)

print(response["choices"][0]["message"]["content"])

# 清理 GPU 缓存
client.clear_cache()
```

### 使用工厂函数

```python
from llm_gemma_transfomers import create_gemma_transformers_client

# 简化创建过程
client = create_gemma_transformers_client(
    use_4bit_quantization=True
)
```

## API 接口

两种实现都提供相同的 API 接口，兼容 OpenAI ChatCompletion 格式：

### `create()` 方法

```python
response = client.create(
    messages,                    # 消息列表
    temperature=0.7,            # 温度参数 (0.0-1.0)
    max_tokens=2048,            # 最大生成 token 数
    top_p=1.0,                  # Top-p 采样
    model=None,                 # 模型名称（可选）
    **kwargs                    # 其他参数
)
```

### 响应格式

```python
{
    "id": "chatcmpl-xxxxx",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "模型名称",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "模型响应内容"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 50,
        "total_tokens": 70
    }
}
```

### 消息格式

```python
messages = [
    {"role": "system", "content": "你是一个友善的AI助手。"},
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回复"},
    {"role": "user", "content": "新的用户问题"}
]
```

## 配置选项

### MLX 版本配置

- `model_name`: MLX 模型名称（默认: `"mlx-community/gemma-3-12b-it-4bit-DWQ"`）

### Transformers 版本配置

- `model_name`: Hugging Face 模型名称（默认: `"google/gemma-3-12b-it"`）
- `use_4bit_quantization`: 是否启用 4bit 量化（默认: `False`）
- `device_map`: 设备映射策略（默认: `"auto"`）
- `trust_remote_code`: 是否信任远程代码（默认: `True`）

## 4bit 量化

Transformers 版本支持 4bit 量化以节省内存：

```python
# 启用 4bit 量化
client = GemmaTransformersChatCompletion(
    use_4bit_quantization=True
)
```

量化配置：
- 使用 NF4 量化类型
- Float16 计算数据类型
- 启用双重量化

## 性能对比

| 实现方式 | 平台支持 | 内存使用 | 推理速度 | 特点 |
|---------|----------|----------|----------|------|
| MLX | Apple Silicon | 低 | 快 | 原生优化，预量化模型 |
| Transformers | 通用 | 可配置 | 中等 | 灵活配置，支持量化 |
| Transformers + 4bit | 通用 | 很低 | 稍慢 | 极低内存占用 |

## 错误处理

两种实现都包含详细的错误处理：

```python
try:
    client = GemmaTransformersChatCompletion()
    response = client.create(messages=messages)
except Exception as e:
    print(f"模型调用失败: {e}")
```

常见错误：
- 模型下载失败：检查网络连接和模型名称
- 内存不足：尝试启用 4bit 量化
- 依赖缺失：确保安装了必要的库

## 模型信息

获取模型详细信息：

```python
info = client.get_model_info()
print(info)
```

返回信息包括：
- 模型名称
- 实现类型（MLX 或 Transformers）
- 量化状态
- 设备信息
- 参数数量（如果可用）

## 运行示例

```bash
cd src/podcast_transcribe/llm
python example_usage.py
```

示例脚本提供：
- 单独测试 MLX 版本
- 单独测试 Transformers 版本（有/无量化）
- 工厂函数测试
- 响应对比测试

## 注意事项

1. **首次使用**：模型会自动下载，可能需要一些时间
2. **内存要求**：
   - 无量化：需要 24GB+ 内存
   - 4bit 量化：需要 8GB+ 内存
3. **Apple Silicon**：推荐使用 MLX 版本以获得最佳性能
4. **其他平台**：推荐使用 Transformers 版本，根据硬件情况选择是否启用量化

## 扩展

基类 `BaseGemmaChatCompletion` 提供了扩展框架，可以轻松添加新的实现：

```python
from base_gemma import BaseGemmaChatCompletion

class MyGemmaImplementation(BaseGemmaChatCompletion):
    def _load_model_and_tokenizer(self):
        # 实现模型加载
        pass
    
    def _generate_response(self, prompt_str, temperature, max_tokens, top_p, **kwargs):
        # 实现响应生成
        pass
```

## 许可证

遵循项目主许可证。 