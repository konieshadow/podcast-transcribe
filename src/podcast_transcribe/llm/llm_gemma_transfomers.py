import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional, Union, Literal
import os

# 禁用 PyTorch 编译以避免在 Gradio Spaces 中的兼容性问题
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# 如果 torch._dynamo 可用，禁用它
try:
    import torch._dynamo
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

from .llm_base import TransformersBaseChatCompletion


class GemmaTransformersChatCompletion(TransformersBaseChatCompletion):
    """基于 Transformers 库的 Gemma 聊天完成实现"""
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device_map: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Gemma 使用 float16 作为默认数据类型
        super().__init__(
            model_name=model_name,
            device_map=device_map,
            device=device,
        )
    
    def _print_error_hints(self):
        """打印Gemma特定的错误提示信息"""
        super()._print_error_hints()
        print("Gemma 特殊要求:")
        print("- 建议使用 Transformers >= 4.21.0")
        print("- 推荐使用 float16 数据类型")
        print("- 确保有足够的GPU内存")


# 为了保持向后兼容性，也可以提供一个简化的工厂函数
def create_gemma_transformers_client(
    model_name: str = "google/gemma-3-4b-it",
    device: Optional[str] = None,
    **kwargs
) -> GemmaTransformersChatCompletion:
    """
    创建 Gemma Transformers 客户端的工厂函数
    
    Args:
        model_name: 模型名称
        device: 指定设备 ("cpu", "cuda", "mps", 等)
        **kwargs: 其他传递给构造函数的参数
    
    Returns:
        GemmaTransformersChatCompletion 实例
    """
    return GemmaTransformersChatCompletion(
        model_name=model_name,
        device=device,
        **kwargs
    )
