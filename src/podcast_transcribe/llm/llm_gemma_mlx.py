from typing import Dict, Union
from .llm_base import BaseChatCompletion


class GemmaMLXChatCompletion(BaseChatCompletion):
    """基于 MLX 库的 Gemma 聊天完成实现"""
    
    def __init__(self, model_name: str = "mlx-community/gemma-3-12b-it-4bit-DWQ"):
        super().__init__(model_name)
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """加载 MLX 模型和分词器"""
        try:
            from mlx_lm import load

            print(f"正在加载 MLX 模型: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            print(f"MLX 模型 {self.model_name} 加载成功")
        except Exception as e:
            print(f"加载模型 {self.model_name} 时出错: {e}")
            print("请确保模型名称正确且可访问。")
            print("您可以尝试使用 'mlx_lm.utils.get_model_path(model_name)' 搜索可用的模型。")
            raise

    def _generate_response(
        self,
        prompt_str: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> str:
        """使用 MLX 生成响应"""
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        
        # 为temperature和top_p创建一个采样器
        sampler = make_sampler(temp=temperature, top_p=top_p)

        # 生成响应
        # mlx_lm中的`generate`函数接受模型、分词器、提示和其他生成参数。
        # 我们需要将我们的参数映射到`generate`期望的参数。
        # `mlx_lm.generate` 的 verbose 参数可用于调试。
        # `temperature` 是 `mlx_lm.generate` 中温度的参数名称。
        response_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt_str,
            max_tokens=max_tokens,
            sampler=sampler,
            # verbose=True # 取消注释以调试生成过程
        )
        
        return response_text

    def get_model_info(self) -> Dict[str, Union[str, bool, int]]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_type": "mlx",
            "library": "mlx_lm"
        }

