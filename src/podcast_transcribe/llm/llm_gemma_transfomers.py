import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional, Union, Literal
from .llm_base import TransformersBaseChatCompletion


class GemmaTransformersChatCompletion(TransformersBaseChatCompletion):
    """基于 Transformers 库的 Gemma 聊天完成实现"""
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        use_4bit_quantization: bool = False,
        device_map: Optional[str] = "auto",
        device: Optional[str] = None,
        trust_remote_code: bool = True
    ):
        # Gemma 使用 float16 作为默认数据类型
        super().__init__(
            model_name=model_name,
            use_4bit_quantization=use_4bit_quantization,
            device_map=device_map,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16
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
    use_4bit_quantization: bool = False,
    device: Optional[str] = None,
    **kwargs
) -> GemmaTransformersChatCompletion:
    """
    创建 Gemma Transformers 客户端的工厂函数
    
    Args:
        model_name: 模型名称
        use_4bit_quantization: 是否使用4bit量化
        device: 指定设备 ("cpu", "cuda", "mps", 等)
        **kwargs: 其他传递给构造函数的参数
    
    Returns:
        GemmaTransformersChatCompletion 实例
    """
    return GemmaTransformersChatCompletion(
        model_name=model_name,
        use_4bit_quantization=use_4bit_quantization,
        device=device,
        **kwargs
    )
