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
        try:
            # 确保消息格式正确
            formatted_messages = []
            for msg in messages:
                if msg.get("role") and msg.get("content"):
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # 使用官方聊天模板
            prompt_str = self.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 调试信息
            print(f"使用官方聊天模板格式化成功，长度: {len(prompt_str)}")
            return prompt_str
            
        except Exception as e:
            print(f"官方聊天模板失败: {e}，使用手动格式化")
            
            # 手动格式化 - 改进版本
            prompt_parts = []
            
            # 处理系统消息 - Gemma 3 的正确处理方式
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            other_messages = [msg for msg in messages if msg.get("role") != "system"]
            
            # 对于 Gemma，系统消息通常需要特殊处理
            if system_messages:
                # 将系统消息作为第一个用户消息的前缀
                system_content = "\n".join([msg["content"] for msg in system_messages])
                if other_messages and other_messages[0].get("role") == "user":
                    # 将系统提示合并到第一个用户消息中
                    first_user_msg = other_messages[0]
                    combined_content = f"{system_content}\n\n{first_user_msg['content']}"
                    other_messages[0] = {"role": "user", "content": combined_content}
                else:
                    # 如果没有用户消息，创建一个包含系统提示的用户消息
                    other_messages.insert(0, {"role": "user", "content": system_content})
            
            # 格式化其他消息
            for message in other_messages:
                role = message.get("role")
                content = message.get("content", "").strip()
                
                if not content:
                    continue
                    
                if role == "user":
                    prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                elif role == "assistant":
                    prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
            
            # 添加生成提示
            prompt_parts.append("<start_of_turn>model")
            
            formatted_prompt = "\n".join(prompt_parts)
            print(f"手动格式化完成，长度: {len(formatted_prompt)}")
            return formatted_prompt

    def _post_process_response(self, response_text: str, prompt_str: str) -> str:
        """
        后处理生成的响应文本，清理提示和特殊标记
        """
        print(f"原始响应长度: {len(response_text)}")
        print(f"原始响应前100字符: {response_text[:100]}")
        
        # 如果模型输出包含提示，则移除提示部分
        if response_text.startswith(prompt_str):
            assistant_message_content = response_text[len(prompt_str):].strip()
            print("检测到响应包含提示，已移除提示部分")
        else:
            # 尝试找到最后一个 "<start_of_turn>model" 标记
            model_start_marker = "<start_of_turn>model"
            if model_start_marker in response_text:
                parts = response_text.split(model_start_marker)
                assistant_message_content = parts[-1].strip()
                print("通过 <start_of_turn>model 标记分割响应")
            else:
                # 如果没有找到标记，使用整个响应
                assistant_message_content = response_text.strip()
                print("未找到特殊标记，使用完整响应")
        
        # 清理结束标记
        end_markers = ["<end_of_turn>", "<|endoftext|>", "</s>"]
        for marker in end_markers:
            if marker in assistant_message_content:
                assistant_message_content = assistant_message_content.split(marker)[0].strip()
                print(f"移除结束标记: {marker}")
        
        # 特殊处理：如果响应看起来像 JSON，尝试提取 JSON 部分
        if "{" in assistant_message_content and "}" in assistant_message_content:
            # 尝试提取 JSON 对象
            first_brace = assistant_message_content.find("{")
            last_brace = assistant_message_content.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                potential_json = assistant_message_content[first_brace:last_brace + 1]
                # 验证是否为有效 JSON
                try:
                    import json
                    json.loads(potential_json)
                    assistant_message_content = potential_json
                    print("提取并验证了 JSON 响应")
                except json.JSONDecodeError:
                    print("JSON 验证失败，保持原始响应")
                    # 如果JSON验证失败，尝试清理常见的非JSON内容
                    lines = assistant_message_content.split('\n')
                    cleaned_lines = []
                    json_started = False
                    
                    for line in lines:
                        line = line.strip()
                        # 跳过明显的解释性文本
                        if any(phrase in line.lower() for phrase in [
                            'here is', 'here\'s', 'based on', 'analysis', 'looking at',
                            'the json', 'response:', 'result:', 'output:', 'answer:',
                            'i can see', 'it appears', 'according to'
                        ]):
                            continue
                        
                        # 如果遇到JSON开始，标记开始
                        if '{' in line:
                            json_started = True
                        
                        # 如果已经开始JSON，保留所有内容
                        if json_started:
                            cleaned_lines.append(line)
                        
                        # 如果遇到JSON结束，停止
                        if '}' in line and json_started:
                            break
                    
                    if cleaned_lines:
                        assistant_message_content = '\n'.join(cleaned_lines)
                        print("清理了非JSON解释性内容")
        
        # 最终清理：移除常见的解释性前缀和后缀
        prefixes_to_remove = [
            "Here is the JSON:", "Here's the JSON:", "The JSON response is:",
            "Based on the analysis:", "Looking at the information:",
            "Here is my analysis:", "The result is:", "My response:",
            "Output:", "Answer:", "Result:"
        ]
        
        for prefix in prefixes_to_remove:
            if assistant_message_content.lower().startswith(prefix.lower()):
                assistant_message_content = assistant_message_content[len(prefix):].strip()
                print(f"移除前缀: {prefix}")
                break
        
        # 移除常见的后缀解释
        suffixes_to_remove = [
            "This JSON object maps each speaker ID to their identified name or role.",
            "Each speaker has been identified based on the provided information.",
            "The identification is based on the dialogue samples and metadata."
        ]
        
        for suffix in suffixes_to_remove:
            if assistant_message_content.lower().endswith(suffix.lower()):
                assistant_message_content = assistant_message_content[:-len(suffix)].strip()
                print(f"移除后缀: {suffix}")
                break
        
        # 最终清理
        assistant_message_content = assistant_message_content.strip()
        
        print(f"处理后响应长度: {len(assistant_message_content)}")
        print(f"处理后响应前100字符: {assistant_message_content[:100]}")
        
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
                print(f"使用设备映射: {self.device_map}")
        
        # 加载模型
        print("开始加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # MPS 或手动设备管理
        if self.device_map is None and self.device is not None:
            print(f"手动移动模型到设备: {self.device}")
            self.model = self.model.to(self.device)
        
        # 设置模型为评估模式
        self.model.eval()
        
        print(f"模型 {self.model_name} 加载成功")
        print(f"模型数据类型: {self.model.dtype}")
        print(f"模型设备: {next(self.model.parameters()).device}")
    
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
        if self.device_map is None or (self.device and hasattr(self.device, 'type') and self.device.type == "mps"):
            inputs = inputs.to(self.device)
        
        # 优化的生成参数配置
        generation_config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,  # 启用 KV 缓存以提高速度
        }
        
        # 温度和采样配置 - 修复 CUDA 采样错误
        if temperature > 0:
            # 确保温度值在合理范围内
            temperature = max(0.01, min(temperature, 2.0))
            top_p = max(0.01, min(top_p, 1.0))
            
            generation_config.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": kwargs.get("top_k", 10),  # 降低top_k以提高确定性
            })
        else:
            # 贪婪解码 - 完全确定性
            generation_config.update({
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            })
        
        # 如果明确指定do_sample=False，强制使用贪婪解码
        if kwargs.get("do_sample") is False:
            generation_config.update({
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            })
            print("强制使用贪婪解码模式")
        
        # 重复惩罚配置 - 针对结构化输出优化
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)  # 默认不使用重复惩罚
        if repetition_penalty != 1.0:
            repetition_penalty = max(1.0, min(repetition_penalty, 1.3))  # 限制在更小范围内
            generation_config["repetition_penalty"] = repetition_penalty
        
        # 移除no_repeat_ngram_size以避免干扰JSON格式
        # generation_config["no_repeat_ngram_size"] = kwargs.get("no_repeat_ngram_size", 2)
        
        # 长度惩罚（可选）- 针对结构化输出调整
        if kwargs.get("length_penalty") and kwargs["length_penalty"] != 1.0:
            length_penalty = max(0.9, min(kwargs["length_penalty"], 1.1))  # 更保守的长度惩罚
            generation_config["length_penalty"] = length_penalty
        
        # 针对结构化输出的特殊配置
        if max_tokens <= 256:  # 如果是短输出任务（如JSON），使用更确定性的配置
            generation_config.update({
                "early_stopping": True,
                "num_beams": 1,  # 使用贪婪搜索
            })
            # 如果允许采样，使用较低的温度
            if generation_config.get("do_sample", False):
                generation_config["temperature"] = min(generation_config.get("temperature", 0.1), 0.3)
                generation_config["top_p"] = min(generation_config.get("top_p", 0.3), 0.5)
            print("检测到短输出任务，使用优化的生成配置")
        
        # 处理stop tokens
        stop_strings = kwargs.get("stop", [])
        if stop_strings:
            # 将stop字符串转换为token IDs
            stop_token_ids = []
            for stop_str in stop_strings:
                try:
                    # 编码stop字符串为token IDs
                    stop_tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_token_ids.extend(stop_tokens)
                except Exception as e:
                    print(f"无法编码stop字符串 '{stop_str}': {e}")
            
            if stop_token_ids:
                # 去重并添加到eos_token_id列表中
                existing_eos = generation_config.get("eos_token_id", self.tokenizer.eos_token_id)
                if isinstance(existing_eos, int):
                    existing_eos = [existing_eos]
                elif existing_eos is None:
                    existing_eos = []
                
                all_stop_tokens = list(set(existing_eos + stop_token_ids))
                generation_config["eos_token_id"] = all_stop_tokens
                print(f"添加了 {len(stop_token_ids)} 个stop token IDs")
        
        # 调试信息
        print(f"生成配置: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
        print(f"输入长度: {len(inputs[0])} tokens")
        
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
            
            print(f"生成完成，输出长度: {len(generated_tokens)} tokens")
            return generated_text
            
        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(f"CUDA 错误，尝试使用 CPU 进行推理: {e}")
                # 尝试移动到 CPU 并重试
                try:
                    inputs_cpu = inputs.cpu()
                    model_cpu = self.model.cpu()
                    
                    with torch.no_grad():
                        outputs = model_cpu.generate(
                            inputs_cpu,
                            **generation_config
                        )
                    
                    # 移回原设备
                    self.model = self.model.to(self.device)
                    
                    generated_tokens = outputs[0][len(inputs_cpu[0]):]
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    print(f"CPU 推理完成，输出长度: {len(generated_tokens)} tokens")
                    return generated_text
                    
                except Exception as cpu_e:
                    print(f"CPU 推理也失败: {cpu_e}")
                    raise e
            else:
                raise e
        except Exception as e:
            print(f"生成响应时出错: {e}")
            import traceback
            traceback.print_exc()
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