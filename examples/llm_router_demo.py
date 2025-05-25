#!/usr/bin/env python3
"""
LLM路由器功能演示
展示重构后的LLM调用系统的各种功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.podcast_transcribe.llm.llm_router import (
    chat_completion, 
    reasoning_completion,
    get_available_providers, 
    get_provider_info,
    get_model_info,
    clear_cache
)


def demo_available_providers():
    """演示获取可用的provider"""
    print("=== 可用的LLM提供者 ===")
    providers = get_available_providers()
    for provider, description in providers.items():
        print(f"- {provider}: {description}")
    print()


def demo_provider_info():
    """演示获取provider详细信息"""
    print("=== Provider详细信息 ===")
    providers = ["mlx", "gemma-transformers", "phi4-transformers"]
    
    for provider in providers:
        try:
            info = get_provider_info(provider)
            print(f"\n{provider}:")
            print(f"  模块路径: {info['module_path']}")
            print(f"  类名: {info['class_name']}")
            print(f"  默认模型: {info['default_model']}")
            print(f"  支持参数: {', '.join(info['supported_params'])}")
        except Exception as e:
            print(f"  获取信息失败: {e}")
    print()


def demo_basic_chat():
    """演示基本聊天功能"""
    print("=== 基本聊天功能演示 ===")
    
    messages = [
        {"role": "user", "content": "你好，请简单介绍一下自己"}
    ]
    
    # 演示不同的provider
    providers_to_test = [
        ("mlx", {}),
        ("gemma-transformers", {"use_4bit_quantization": False}),
        # 注意：实际使用时需要确保模型可用
        # ("phi4-transformers", {"use_4bit_quantization": False}),
    ]
    
    for provider, kwargs in providers_to_test:
        try:
            print(f"\n--- 使用 {provider} ---")
            response = chat_completion(
                messages=messages,
                provider=provider,
                max_tokens=100,
                temperature=0.7,
                **kwargs
            )
            
            print(f"响应: {response['choices'][0]['message']['content']}")
            print(f"Token使用: {response['usage']}")
            
        except Exception as e:
            print(f"使用 {provider} 时出错: {e}")
    
    print()


def demo_reasoning_tasks():
    """演示推理任务功能"""
    print("=== 推理任务功能演示 ===")
    
    # 数学推理任务
    math_messages = [
        {"role": "user", "content": "解这个方程：2x + 5 = 15，请详细说明解题步骤"}
    ]
    
    # 逻辑推理任务
    logic_messages = [
        {"role": "user", "content": "如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？请分析这个逻辑问题"}
    ]
    
    reasoning_tasks = [
        ("数学推理", math_messages),
        ("逻辑推理", logic_messages)
    ]
    
    for task_name, messages in reasoning_tasks:
        try:
            print(f"\n--- {task_name} ---")
            
            # 注意：实际使用时需要确保Phi-4模型可用
            # 这里使用gemma-transformers作为演示
            response = chat_completion(
                messages=messages,
                provider="gemma-transformers",  # 实际应该使用 "phi4-transformers"
                temperature=0.3,
                max_tokens=500
            )
            
            print(f"响应: {response['choices'][0]['message']['content']}")
            print(f"Token使用: {response['usage']}")
            
            # 如果使用phi4-transformers，可以使用专门的推理接口
            # response = reasoning_completion(
            #     messages=messages,
            #     provider="phi4-transformers",
            #     extract_reasoning_steps=True
            # )
            # if "reasoning_steps" in response:
            #     print("推理步骤:")
            #     for i, step in enumerate(response["reasoning_steps"], 1):
            #         print(f"  {i}. {step['title']}: {step['content'].strip()}")
            
        except Exception as e:
            print(f"推理任务 {task_name} 出错: {e}")
    
    print()


def demo_model_info():
    """演示获取模型信息"""
    print("=== 模型信息演示 ===")
    
    providers_to_check = [
        ("mlx", {}),
        ("gemma-transformers", {"use_4bit_quantization": False}),
        # ("phi4-transformers", {"use_4bit_quantization": False}),
    ]
    
    for provider, kwargs in providers_to_check:
        try:
            print(f"\n--- {provider} 模型信息 ---")
            info = get_model_info(provider, **kwargs)
            
            for key, value in info.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"获取 {provider} 模型信息时出错: {e}")
    
    print()


def demo_advanced_features():
    """演示高级功能"""
    print("=== 高级功能演示 ===")
    
    # 多轮对话
    print("\n--- 多轮对话 ---")
    conversation = [
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。"},
        {"role": "user", "content": "能给我举个具体的例子吗？"}
    ]
    
    try:
        response = chat_completion(
            messages=conversation,
            provider="gemma-transformers",
            temperature=0.8,
            max_tokens=200
        )
        
        print(f"多轮对话响应: {response['choices'][0]['message']['content']}")
        print(f"Token使用: {response['usage']}")
        
    except Exception as e:
        print(f"多轮对话出错: {e}")
    
    # 自定义参数
    print("\n--- 自定义参数 ---")
    try:
        response = chat_completion(
            messages=[{"role": "user", "content": "写一首关于春天的短诗"}],
            provider="gemma-transformers",
            temperature=0.9,  # 更高的创造性
            max_tokens=150,
            top_p=0.9
        )
        
        print(f"创意写作响应: {response['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"创意写作出错: {e}")
    
    print()


def demo_error_handling():
    """演示错误处理"""
    print("=== 错误处理演示 ===")
    
    # 测试不支持的provider
    try:
        chat_completion(
            messages=[{"role": "user", "content": "测试"}],
            provider="unsupported_provider"
        )
    except Exception as e:
        print(f"不支持的provider错误: {e}")
    
    # 测试无效参数
    try:
        get_provider_info("nonexistent_provider")
    except Exception as e:
        print(f"无效provider信息错误: {e}")
    
    print()


def main():
    """主函数"""
    print("LLM路由器功能演示")
    print("=" * 50)
    
    try:
        # 基础信息演示
        demo_available_providers()
        demo_provider_info()
        
        # 功能演示
        demo_basic_chat()
        demo_reasoning_tasks()
        demo_model_info()
        demo_advanced_features()
        
        # 错误处理演示
        demo_error_handling()
        
        # 清理缓存
        print("=== 清理缓存 ===")
        clear_cache()
        print("缓存已清理")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
    finally:
        print("\n演示结束")


if __name__ == "__main__":
    main() 