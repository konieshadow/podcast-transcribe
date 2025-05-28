import time
import uuid
import torch
from typing import List, Dict, Optional, Union, Literal
from abc import ABC, abstractmethod


class BaseChatCompletion(ABC):
    """Gemma 聊天完成的基类，包含公共功能"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def _load_model_and_tokenizer(self):
        """加载模型和分词器的抽象方法，由子类实现"""
        pass
    
    @abstractmethod
    def _generate_response(self, prompt_str: str, temperature: float, max_tokens: int, top_p: float, **kwargs) -> str:
        """生成响应的抽象方法，由子类实现"""
        pass

    def _format_messages_for_gemma(self, messages: List[Dict[str, str]]) -> str:
        """
        为Gemma格式化消息。
        Gemma期望特定的格式，通常类似于：
        <start_of_turn>user
        {user_message}<end_of_turn>
        <start_of_turn>model
        {assistant_message}<end_of_turn>
        ...
        <start_of_turn>user
        {current_user_message}<end_of_turn>
        <start_of_turn>model
        """
        # 尝试使用分词器的聊天模板（如果可用）
        try:
            # Hugging Face分词器中的apply_chat_template方法
            # 通常需要一个字典列表，每个字典包含'role'和'content'。
            # 我们需要确保我们的`messages`格式兼容。
            # add_generation_prompt=True 至关重要，以确保模型知道轮到它发言了。
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # 如果apply_chat_template失败或不可用，则回退到手动格式化
            prompt_parts = []
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if role == "user":
                    prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                elif role == "assistant":
                    prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
                elif role == "system": # Gemma可能不会以相同的方式显式使用'system'，通常是前置的
                    # 对于Gemma，系统提示通常只是前置到第一个用户消息或隐式处理。
                    # 我们会在这里前置它，尽管其有效性取决于特定的Gemma微调。
                    # 一种常见的模式是在开头放置系统指令，不使用特殊标记。
                    # 然而，为了保持结构化，我们将尝试一种通用方法。
                    # 如果分词器在其模板中有特定的方式来处理系统提示，
                    # 那么`apply_chat_template`将是首选。
                    # 由于我们处于回退状态，这是一个最佳猜测。
                    # 一些模型期望系统提示在轮次结构之外，或者在最开始。
                    # 为了在回退中简化，我们只做前置处理。
                    # 如果`apply_chat_template`不可用，更健壮的解决方案是检查模型的特定聊天模板。
                    prompt_parts.insert(0, f"<start_of_turn>system\n{content}<end_of_turn>")

            # 添加提示，让模型开始生成
            prompt_parts.append("<start_of_turn>model")
            return "\n".join(prompt_parts)

    def _post_process_response(self, response_text: str, prompt_str: str) -> str:
        """
        后处理生成的响应文本，清理提示和特殊标记
        """
        # 后处理：Gemma的输出可能包含输入提示或特殊标记。
        # 我们需要清理这些，以仅返回助手的最新消息。
        # 一种常见的模式是，模型输出将以我们给它的提示开始，
        # 或者它可能包含 <start_of_turn>model 标记，然后是其响应。

        # 如果模型输出包含提示，然后是新的响应：
        if response_text.startswith(prompt_str):
            assistant_message_content = response_text[len(prompt_str):].strip()
        else:
            # 如果模型不回显提示，则可能需要更复杂的清理。
            # 对于Gemma，响应通常跟随提示的最后一部分 "<start_of_turn>model\n"。
            # 让我们尝试找到最后一个 "<start_of_turn>model" 并获取其后的文本。
            # 这是一种启发式方法，可能需要根据实际模型输出进行调整。
            parts = response_text.split("<start_of_turn>model")
            if len(parts) > 1:
                assistant_message_content = parts[-1].strip()
                # 进一步清理 <end_of_turn> 或其他特殊标记
                assistant_message_content = assistant_message_content.split("<end_of_turn>")[0].strip()
            else: # 如果上述方法不起作用，则回退
                assistant_message_content = response_text.strip()
        
        return assistant_message_content

    def _calculate_tokens(self, prompt_str: str, assistant_message_content: str) -> Dict[str, int]:
        """
        计算token数量（近似值，因为确切的OpenAI分词可能不同）
        """
        # 对于提示token，我们对输入到模型的字符串进行分词。
        # 对于完成token，我们对生成的助手消息进行分词。
        prompt_tokens = len(self.tokenizer.encode(prompt_str))
        completion_tokens = len(self.tokenizer.encode(assistant_message_content))
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

    def _build_chat_completion_response(self, assistant_message_content: str, token_usage: Dict[str, int]) -> Dict:
        """
        构建模仿OpenAI结构的响应对象
        基于: https://platform.openai.com/docs/api-reference/chat/object
        """
        # 获取完成的当前时间戳
        created_timestamp = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}" # 创建一个唯一的ID

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_timestamp,
            "model": self.model_name, # 报告我们使用的模型名称
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_message_content,
                    },
                    "finish_reason": "stop", # 假定为 "stop"
                }
            ],
            "usage": token_usage,
        }

    def create(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs,
    ):
        """
        创建聊天完成响应。
        模仿OpenAI的ChatCompletion.create方法。
        """
        # 为Gemma格式化消息
        prompt_str = self._format_messages_for_gemma(messages)

        # 生成响应（由子类实现）
        response_text = self._generate_response(prompt_str, temperature, max_tokens, top_p, **kwargs)
        
        # 后处理响应
        assistant_message_content = self._post_process_response(response_text, prompt_str)
        
        # 计算token使用量
        token_usage = self._calculate_tokens(prompt_str, assistant_message_content)
        
        # 构建响应对象
        return self._build_chat_completion_response(assistant_message_content, token_usage)


class TransformersBaseChatCompletion(BaseChatCompletion):
    """基于Transformers库的聊天完成基类，提供通用的设备管理和量化功能"""
    
    def __init__(
        self,
        model_name: str,
        device_map: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_name)
        torch.set_float32_matmul_precision('high') # 设置 TensorFloat32 精度
        self.device_map = device_map
        self.device = device
        
        # 加载模型和分词器
        self._load_model_and_tokenizer()
    
    def _load_tokenizer(self):
        """加载分词器"""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("请先安装 transformers 库: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # 设置 pad_token 如果不存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("请先安装 transformers 库: pip install transformers")
        
        print(f"正在加载模型: {self.model_name}")
        print(f"目标设备: {self.device}")
        print(f"设备映射: {self.device_map}")
        
        # 配置模型加载参数
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # 处理设备映射
        if self.device_map is not None:
            if self.device and self.device.type == "mps":
                print("警告: MPS 设备不支持 device_map，将手动管理设备")
            else:
                model_kwargs["device_map"] = self.device_map
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # MPS 或手动设备管理
        if self.device_map is None:
            print(f"手动移动模型到设备: {self.device}")
            self.model = self.model.to(self.device)
        
        print(f"模型 {self.model_name} 加载成功")
    
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        try:
            self._load_tokenizer()
            self._load_model()
        except Exception as e:
            print(f"加载模型 {self.model_name} 时出错: {e}")
            self._print_error_hints()
            raise
    
    def _print_error_hints(self):
        """打印错误提示信息"""
        print("请确保模型名称正确且可访问。")
        if self.device and self.device == "mps":
            print("MPS 设备注意事项:")
            print("- 不支持 device_map")
            print("- 确保 PyTorch 版本支持 MPS")
    
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
        if self.device_map is None or (self.device and self.device.type == "mps"):
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
            
            # 解码生成的文本，跳过输入部分
            generated_tokens = outputs[0][len(inputs[0]):]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Union[str, bool, int]]:
        """获取模型信息"""
        model_info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "device_type": self.device.type,
            "device_map": self.device_map,
            "model_type": "transformers",
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cuda_available": torch.cuda.is_available(),
        }
        
        # 添加模型配置信息（如果可用）
        try:
            if hasattr(self.model, "config"):
                config = self.model.config
                model_info.update({
                    "vocab_size": getattr(config, "vocab_size", "未知"),
                    "hidden_size": getattr(config, "hidden_size", "未知"),
                    "num_layers": getattr(config, "num_hidden_layers", "未知"),
                    "num_attention_heads": getattr(config, "num_attention_heads", "未知"),
                })
        except Exception:
            pass
        
        return model_info
    
    def clear_cache(self):
        """清理 GPU 缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU 缓存已清理") 