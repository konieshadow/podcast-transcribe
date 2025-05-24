
# 添加项目根目录到Python路径
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from podcast_transcribe.llm.llm_gemma_mlx import GemmaMLXChatCompletion


if __name__ == "__main__":
    # 示例用法：
    print("正在初始化 Gemma MLX 聊天补全...")
    try:
        # gemma_chat = GemmaMLXChatCompletion(model_name="mlx-community/gemma-3-12b-it-4bit-DWQ")
        # 或者，如果您有更小、更快的模型，可以尝试使用，例如："mlx-community/gemma-2b-it-8bit"
        gemma_chat = GemmaMLXChatCompletion(model_name="mlx-community/gemma-3-12b-it-4bit-DWQ")
        print("模型加载成功。")

        print("\n--- 示例 1: 简单用户查询 ---")
        messages_example1 = [
            {"role": "user", "content": "你好，你是谁？"}
        ]
        response1 = gemma_chat.create(messages=messages_example1, max_tokens=50)
        print("响应 1:")
        print(f"  助手: {response1['choices'][0]['message']['content']}")
        print(f"  用量: {response1['usage']}")

        print("\n--- 示例 2: 带历史记录的对话 ---")
        messages_example2 = [
            {"role": "user", "content": "法国的首都是哪里？"},
            {"role": "assistant", "content": "法国的首都是巴黎。"},
            {"role": "user", "content": "你能告诉我一个关于它的有趣事实吗？"}
        ]
        response2 = gemma_chat.create(messages=messages_example2, max_tokens=100, temperature=0.8)
        print("响应 2:")
        print(f"  助手: {response2['choices'][0]['message']['content']}")
        print(f"  用量: {response2['usage']}")
        
        print("\n--- 示例 3: 系统提示 (实验性，效果取决于模型微调) ---")
        messages_example3 = [
            {"role": "system", "content": "你是一位富有诗意的助手，擅长用富有创意的方式解释复杂的编程概念。"},
            {"role": "user", "content": "解释一下编程中递归的概念。"}
        ]
        response3 = gemma_chat.create(messages=messages_example3, max_tokens=150)
        print("响应 3:")
        print(f"  助手: {response3['choices'][0]['message']['content']}")
        print(f"  用量: {response3['usage']}")

        print("\n--- 示例 4: 使用 max_tokens 强制缩短响应 ---")
        messages_example4 = [
            {"role": "user", "content": "给我讲一个关于勇敢骑士的很长的故事。"}
        ]
        response4 = gemma_chat.create(messages=messages_example4, max_tokens=20) # 非常短
        print("响应 4:")
        print(f"  助手: {response4['choices'][0]['message']['content']}")
        print(f"  用量: {response4['usage']}")
        if response4['usage']['completion_tokens'] >= 20:
             print("  注意：由于 max_tokens，补全可能已被截断。")


    except Exception as e:
        print(f"示例用法期间发生错误: {e}")
        import traceback
        traceback.print_exc()
