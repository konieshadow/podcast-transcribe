import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional, Union, Literal
from .llm_gemma_base import BaseGemmaChatCompletion


class GemmaTransformersChatCompletion(BaseGemmaChatCompletion):
    """基于 Transformers 库的 Gemma 聊天完成实现"""
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        use_4bit_quantization: bool = False,
        device_map: Optional[str] = "auto",
        device: Optional[str] = None,
        trust_remote_code: bool = True
    ):
        super().__init__(model_name)
        self.use_4bit_quantization = use_4bit_quantization
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        
        # 智能设备选择
        if device:
            self.device = torch.device(device)
        else:
            # 自动选择最佳设备
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        # 如果指定了具体设备，调整 device_map
        if device and device_map == "auto":
            if device == "mps":
                self.device_map = None  # MPS 不支持 device_map，需要手动管理
            elif device == "cpu":
                self.device_map = "cpu"
            elif device.startswith("cuda"):
                self.device_map = "auto"
        
        # 加载模型和分词器
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """加载 Gemma 模型和分词器"""
        try:
            print(f"正在加载模型: {self.model_name}")
            print(f"4bit量化: {'启用' if self.use_4bit_quantization else '禁用'}")
            print(f"目标设备: {self.device}")
            print(f"设备映射: {self.device_map}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # 设置 pad_token 如果不存在
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 配置模型加载参数
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": torch.float16,  # 使用 float16 以节省内存
            }
            
            # 处理量化配置
            if self.use_4bit_quantization:
                if self.device.type == "mps":
                    print("警告: MPS 设备不支持 4bit 量化，将禁用量化")
                    self.use_4bit_quantization = False
                else:
                    # 配置 4bit 量化
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    print("使用 4bit 量化配置")
            
            # 处理设备映射
            if self.device_map is not None:
                if self.device.type == "mps":
                    print("警告: MPS 设备不支持 device_map，将手动管理设备")
                else:
                    model_kwargs["device_map"] = self.device_map
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # MPS 或手动设备管理
            if self.device_map is None or self.device.type == "mps":
                if not self.use_4bit_quantization:
                    print(f"手动移动模型到设备: {self.device}")
                    self.model = self.model.to(self.device)
            
            print(f"模型 {self.model_name} 加载成功")
            
        except Exception as e:
            print(f"加载模型 {self.model_name} 时出错: {e}")
            print("请确保模型名称正确且可访问。")
            if self.use_4bit_quantization:
                print("如果使用量化，请确保已安装 bitsandbytes 库: pip install bitsandbytes")
            if self.device.type == "mps":
                print("MPS 设备注意事项:")
                print("- 不支持 4bit 量化")
                print("- 不支持 device_map")
                print("- 确保 PyTorch 版本支持 MPS")
            raise
    
    def _generate_response(
        self,
        prompt_str: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> str:
        """使用 transformers 生成响应"""
        
        # 对提示进行编码
        inputs = self.tokenizer.encode(prompt_str, return_tensors="pt")
        
        # 移动输入到正确的设备
        # 如果使用了 device_map="auto" 或量化，模型会自动处理设备
        # 否则需要手动移动输入到模型所在设备
        if self.device_map is None or self.device.type == "mps":
            inputs = inputs.to(self.device)
        
        # 生成参数
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True if temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 3),
        }
        
        # 如果温度为0，使用贪婪解码
        if temperature == 0:
            generation_config["do_sample"] = False
            generation_config.pop("temperature", None)
            generation_config.pop("top_p", None)
        
        try:
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **generation_config
                )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            return generated_text
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Union[str, bool, int]]:
        """获取模型信息"""
        model_info = {
            "model_name": self.model_name,
            "use_4bit_quantization": self.use_4bit_quantization,
            "device": str(self.device),
            "device_type": self.device.type,
            "device_map": self.device_map,
            "model_type": "transformers",
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cuda_available": torch.cuda.is_available(),
        }
        
        # 添加模型参数数量信息（如果可用）
        try:
            if hasattr(self.model, "num_parameters"):
                model_info["num_parameters"] = self.model.num_parameters()
            elif hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
                # 粗略估计参数数量
                vocab_size = self.model.config.vocab_size
                hidden_size = getattr(self.model.config, "hidden_size", 0)
                num_layers = getattr(self.model.config, "num_hidden_layers", 0)
                model_info["vocab_size"] = vocab_size
                model_info["hidden_size"] = hidden_size
                model_info["num_layers"] = num_layers
        except Exception:
            pass
        
        return model_info
    
    def clear_cache(self):
        """清理 GPU 缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU 缓存已清理")


# 为了保持向后兼容性，也可以提供一个简化的工厂函数
def create_gemma_transformers_client(
    model_name: str = "google/gemma-3-12b-it",
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
