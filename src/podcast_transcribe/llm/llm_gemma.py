from mlx_lm import load, generate
from mlx_lm.utils import get_model_path
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer
import time
import uuid
from typing import List, Dict, Optional, Union, Literal


class GemmaMLXChatCompletion:
    def __init__(self, model_name: str = "mlx-community/gemma-3-4b-it-4bit-DWQ"):
        self.model_name = model_name
        try:
            self.model, self.tokenizer = load(model_name)
        except Exception as e:
            print(f"加载模型 {model_name} 时出错: {e}")
            print("请确保模型名称正确且可访问。")
            print("您可以尝试使用 'mlx_lm.utils.get_model_path(model_name)' 搜索可用的模型。")
            raise

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

    def create(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        # stream: bool = False, # mlx_lm.generate 不以相同方式直接支持流式传输
        model: Optional[str] = None, # 允许在每次调用时覆盖模型，尽管对于此类的结构而言不典型
        **kwargs, # 用于我们未使用的其他潜在OpenAI参数
    ):
        """
        创建聊天完成响应。
        模仿OpenAI的ChatCompletion.create方法。
        """
        if model and model != self.model_name:
            # 这是一个简化的处理。在实际场景中，您可能希望加载新模型。
            # 目前，我们将只打印一个警告并使用初始化的模型。
            print(f"警告: 'model' 参数 ({model}) 与初始化的模型 ({self.model_name}) 不同。"
                  f"正在使用初始化的模型。要使用不同的模型，请重新初始化该类。")

        # 为Gemma格式化消息
        # Gemma期望一个字符串提示。我们需要转换消息列表。
        prompt_str = self._format_messages_for_gemma(messages)

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


        # 计算token数量（近似值，因为确切的OpenAI分词可能不同）
        # 对于提示token，我们对输入到模型的字符串进行分词。
        # 对于完成token，我们对生成的助手消息进行分词。
        prompt_tokens = len(self.tokenizer.encode(prompt_str))
        completion_tokens = len(self.tokenizer.encode(assistant_message_content))
        total_tokens = prompt_tokens + completion_tokens

        # 获取完成的当前时间戳
        created_timestamp = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}" # 创建一个唯一的ID

        # 构建模仿OpenAI结构的响应对象
        # 基于: https://platform.openai.com/docs/api-reference/chat/object
        chat_completion_object = {
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
                    "finish_reason": "stop", # mlx_lm.generate 在达到 max_tokens 或 EOS 时停止。
                                            # 如果达到 max_tokens 则为 "length"，如果为 EOS 则为 "stop"。
                                            # 我们没有直接的方法从 mlx_lm.generate 获取此信息，
                                            # 因此我们暂时假定为 "stop"。
                                            # 一种更精确的方法是检查 len(response_tokens) == max_tokens
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            # "system_fingerprint": None, # 此处不适用
        }
        return chat_completion_object