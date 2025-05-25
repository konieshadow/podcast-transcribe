import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional, Union, Literal
from .llm_base import TransformersBaseChatCompletion


class Phi4TransformersChatCompletion(TransformersBaseChatCompletion):
    """基于 Transformers 库的 Phi-4-mini-reasoning 聊天完成实现"""
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-mini-reasoning",
        use_4bit_quantization: bool = False,
        device_map: Optional[str] = "auto",
        device: Optional[str] = None,
        trust_remote_code: bool = True
    ):
        # Phi-4 使用 bfloat16 作为推荐数据类型
        super().__init__(
            model_name=model_name,
            use_4bit_quantization=use_4bit_quantization,
            device_map=device_map,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16
        )
    
    def _print_error_hints(self):
        """打印Phi-4特定的错误提示信息"""
        super()._print_error_hints()
        print("Phi-4 特殊要求:")
        print("- 建议使用 Transformers >= 4.51.3")
        print("- 推荐使用 bfloat16 数据类型")
        print("- 模型支持 128K token 上下文长度")
    
    def _format_phi4_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        格式化消息为 Phi-4 的聊天格式
        Phi-4 使用特定的聊天模板格式
        """
        # 使用 tokenizer 的内置聊天模板
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # 如果没有聊天模板，使用 Phi-4 的标准格式
            formatted_prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    formatted_prompt += f"<|system|>\n{content}<|end|>\n"
                elif role == "user":
                    formatted_prompt += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    formatted_prompt += f"<|assistant|>\n{content}<|end|>\n"
            
            # 添加助手开始标记
            formatted_prompt += "<|assistant|>\n"
            return formatted_prompt
    
    def _generate_response(
        self,
        prompt_str: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        enable_reasoning: bool = True,
        **kwargs
    ) -> str:
        """使用 transformers 生成响应，针对 Phi-4 推理功能优化"""
        
        # 对提示进行编码
        inputs = self.tokenizer.encode(prompt_str, return_tensors="pt")
        
        # 移动输入到正确的设备
        if self.device_map is None or self.device.type == "mps":
            inputs = inputs.to(self.device)
        
        # Phi-4-mini-reasoning 优化的生成参数
        generation_config = {
            "max_new_tokens": min(max_tokens, 32768),  # Phi-4-mini 支持最大 32K token
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True if temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 3),
        }
        
        # 推理模式配置
        if enable_reasoning and "reasoning" in self.model_name.lower():
            # 为推理任务优化的配置
            generation_config.update({
                "temperature": max(temperature, 0.1),  # 推理模式下保持一定的温度
                "top_p": min(top_p, 0.95),  # 推理模式下限制 top_p
                "do_sample": True,  # 推理模式下总是启用采样
                "early_stopping": False,  # 允许完整的推理过程
            })
        
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
    
    def create(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        model: Optional[str] = None,
        enable_reasoning: bool = True,
        **kwargs,
    ):
        """
        创建聊天完成响应，支持Phi-4特有的推理功能
        """
        if model and model != self.model_name:
            print(f"警告: 'model' 参数 ({model}) 与初始化的模型 ({self.model_name}) 不同。"
                  f"正在使用初始化的模型。要使用不同的模型，请重新初始化该类。")

        # 检查是否为推理任务
        is_reasoning_task = self._is_reasoning_task(messages)
        
        # 格式化消息为 Phi-4 聊天格式
        if is_reasoning_task and enable_reasoning:
            prompt_str = self._format_reasoning_prompt(messages)
        else:
            prompt_str = self._format_phi4_messages(messages)

        # 生成响应
        response_text = self._generate_response(
            prompt_str, 
            temperature, 
            max_tokens, 
            top_p, 
            enable_reasoning=enable_reasoning and is_reasoning_task,
            **kwargs
        )
        
        # 后处理响应（使用基类的方法，但针对Phi-4调整）
        assistant_message_content = self._post_process_phi4_response(response_text, prompt_str)
        
        # 计算token使用量
        token_usage = self._calculate_tokens(prompt_str, assistant_message_content)
        
        # 构建响应对象
        response = self._build_chat_completion_response(assistant_message_content, token_usage)
        
        # 添加Phi-4特有的信息
        response["reasoning_enabled"] = enable_reasoning and is_reasoning_task
        
        return response
    
    def _post_process_phi4_response(self, response_text: str, prompt_str: str) -> str:
        """
        后处理Phi-4生成的响应文本
        """
        # Phi-4的输出通常不包含输入提示，直接返回生成的内容
        assistant_message_content = response_text.strip()
        
        # 清理可能的特殊标记
        if assistant_message_content.endswith("<|end|>"):
            assistant_message_content = assistant_message_content[:-7].strip()
        
        return assistant_message_content
    
    def _is_reasoning_task(self, messages: List[Dict[str, str]]) -> bool:
        """检测是否为推理任务"""
        reasoning_keywords = [
            "解题", "推理", "计算", "证明", "分析", "逻辑", "步骤",
            "solve", "reasoning", "calculate", "prove", "analyze", "logic", "step"
        ]
        
        for message in messages:
            content = message.get("content", "").lower()
            if any(keyword in content for keyword in reasoning_keywords):
                return True
        
        return False
    
    def _format_reasoning_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        为推理任务格式化特殊的提示词
        """
        # 添加推理指导的系统消息
        reasoning_system_msg = {
            "role": "system",
            "content": "你是一个专业的数学推理助手。请逐步分析问题，展示详细的推理过程，包括：\n1. 问题理解\n2. 解题思路\n3. 具体步骤\n4. 最终答案\n\n每个步骤都要清晰明了。"
        }
        
        # 将推理系统消息添加到消息列表的开头
        enhanced_messages = [reasoning_system_msg] + messages
        
        # 使用标准格式化方法
        return self._format_phi4_messages(enhanced_messages)
    
    def reasoning_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,  # 推理任务使用较低的温度
        max_tokens: int = 2048,   # 推理任务需要更多 tokens
        top_p: float = 0.9,
        extract_reasoning_steps: bool = True,
        **kwargs
    ) -> Dict[str, Union[str, Dict, List]]:
        """
        专门用于推理任务的聊天完成接口
        
        Args:
            messages: 对话消息列表
            temperature: 采样温度（推理任务建议使用较低值）
            max_tokens: 最大生成token数量
            top_p: top-p采样参数
            extract_reasoning_steps: 是否提取推理步骤
            **kwargs: 其他参数
        
        Returns:
            包含推理步骤的响应字典
        """
        # 强制启用推理模式
        response = self.create(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            enable_reasoning=True,
            **kwargs
        )
        
        if extract_reasoning_steps:
            # 提取推理步骤
            content = response["choices"][0]["message"]["content"]
            reasoning_steps = self._extract_reasoning_steps(content)
            response["reasoning_steps"] = reasoning_steps
        
        return response
    
    def _extract_reasoning_steps(self, content: str) -> List[Dict[str, str]]:
        """
        从响应内容中提取推理步骤
        """
        steps = []
        lines = content.split('\n')
        current_step = {"title": "", "content": ""}
        
        step_patterns = [
            "1. 问题理解", "2. 解题思路", "3. 具体步骤", "4. 最终答案",
            "步骤", "分析", "解答", "结论", "reasoning", "step", "analysis", "solution"
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是新的步骤开始
            is_new_step = any(pattern in line.lower() for pattern in step_patterns)
            if is_new_step and current_step["content"]:
                steps.append(current_step.copy())
                current_step = {"title": line, "content": ""}
            elif is_new_step:
                current_step["title"] = line
            else:
                if current_step["title"]:
                    current_step["content"] += line + "\n"
                else:
                    current_step["content"] = line + "\n"
        
        # 添加最后一个步骤
        if current_step["title"] or current_step["content"]:
            steps.append(current_step)
        
        return steps
    
    def get_model_info(self) -> Dict[str, Union[str, bool, int]]:
        """获取 Phi-4 模型信息"""
        model_info = super().get_model_info()
        
        # 添加Phi-4特有的信息
        model_info.update({
            "model_family": "Phi-4-mini-reasoning",
            "parameters": "3.8B",
            "context_length": "128K tokens",
            "specialization": "数学推理优化",
        })
        
        return model_info


# 工厂函数
def create_phi4_transformers_client(
    model_name: str = "microsoft/Phi-4-mini-reasoning",
    use_4bit_quantization: bool = False,
    device: Optional[str] = None,
    **kwargs
) -> Phi4TransformersChatCompletion:
    """
    创建 Phi-4 Transformers 客户端的工厂函数
    
    Args:
        model_name: 模型名称，默认为 microsoft/Phi-4-mini-reasoning
        use_4bit_quantization: 是否使用4bit量化
        device: 指定设备 ("cpu", "cuda", "mps", 等)
        **kwargs: 其他传递给构造函数的参数
    
    Returns:
        Phi4TransformersChatCompletion 实例
    """
    return Phi4TransformersChatCompletion(
        model_name=model_name,
        use_4bit_quantization=use_4bit_quantization,
        device=device,
        **kwargs
    )

def create_reasoning_client(
    model_name: str = "microsoft/Phi-4-mini-reasoning",
    use_4bit_quantization: bool = False,
    device: Optional[str] = None,
    **kwargs
) -> Phi4TransformersChatCompletion:
    """
    创建专门用于推理任务的 Phi-4 客户端
    
    Args:
        model_name: 模型名称，推荐使用 microsoft/Phi-4-mini-reasoning
        use_4bit_quantization: 是否使用4bit量化
        device: 指定设备 ("cpu", "cuda", "mps", 等)
        **kwargs: 其他传递给构造函数的参数
    
    Returns:
        优化了推理功能的 Phi4TransformersChatCompletion 实例
    """
    # 确保使用推理模型
    if "reasoning" not in model_name.lower():
        print("警告: 建议使用包含 'reasoning' 的模型名称以获得最佳推理性能")
    
    return Phi4TransformersChatCompletion(
        model_name=model_name,
        use_4bit_quantization=use_4bit_quantization,
        device=device,
        **kwargs
    ) 