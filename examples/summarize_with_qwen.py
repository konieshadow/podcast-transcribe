#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Qwen本地模型进行播客摘要的示例脚本
"""

import os
import argparse
import logging
from pathlib import Path

from src.podcast_transcribe.transcriber import PodcastTranscriber
from src.podcast_transcribe.utils.summarization import SummarizerType

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qwen_summarize_example")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='使用Qwen模型摘要播客内容示例'
    )
    
    parser.add_argument(
        '--audio_file',
        type=str,
        required=True,
        help='音频文件路径'
    )
    
    parser.add_argument(
        '--whisper_model_path',
        type=str,
        required=True,
        help='Whisper模型路径'
    )
    
    parser.add_argument(
        '--hf_token',
        type=str,
        required=True,
        help='Hugging Face API令牌'
    )
    
    parser.add_argument(
        '--qwen_model_path',
        type=str,
        default="Qwen/Qwen1.5-1.8B",
        help='Qwen模型路径（默认为Qwen/Qwen1.5-1.8B）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='推理设备（默认为cuda）'
    )
    
    parser.add_argument(
        '--no_4bit_quantization',
        action='store_true',
        help='禁用4bit量化加载模型'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='生成温度，控制随机性（默认为0.7）'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='核采样的概率阈值（默认为0.9）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 提取音频文件名
    audio_basename = os.path.basename(args.audio_file)
    audio_name = os.path.splitext(audio_basename)[0]
    
    # 输出文件路径
    transcript_file = output_dir / f"{audio_name}_transcript.txt"
    summary_file = output_dir / f"{audio_name}_summary.txt"
    
    logger.info(f"初始化播客转录器，使用Qwen模型: {args.qwen_model_path}")
    
    # 初始化转录器
    transcriber = PodcastTranscriber(
        whisper_model_path=args.whisper_model_path,
        hf_token=args.hf_token,
        summarization_model=args.qwen_model_path,
        summarizer_type="qwen",  # 使用Qwen模型
        summarizer_device=args.device,
        summarizer_load_in_4bit=not args.no_4bit_quantization
    )
    
    # 构建摘要参数
    summary_kwargs = {
        'temperature': args.temperature,
        'top_p': args.top_p
    }
    
    logger.info("开始转录和摘要...")
    
    # 执行转录和摘要
    summary, transcript = transcriber.transcribe_and_summarize(
        audio_file=args.audio_file,
        output_transcript_file=transcript_file,
        output_summary_file=summary_file,
        **summary_kwargs
    )
    
    logger.info(f"转录文本已保存到: {transcript_file}")
    logger.info(f"摘要文本已保存到: {summary_file}")
    
    # 打印摘要
    print("\n===== Qwen生成的播客摘要 =====")
    print(summary)
    print("==============================\n")


if __name__ == "__main__":
    main() 