"""
命令行界面模块
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from .transcriber import PodcastTranscriber, TranscriberType
from .utils.summarization import SummarizerType


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    设置日志记录
    
    参数:
        verbose: 是否启用详细日志
        
    返回:
        日志记录器
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # 创建日志目录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 设置日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'podcast_transcribe_{timestamp}.log'
    
    # 配置日志记录器
    logger = logging.getLogger('podcast_transcribe')
    logger.setLevel(level)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    返回:
        解析的参数
    """
    parser = argparse.ArgumentParser(
        description='播客音频转录与摘要工具'
    )
    
    parser.add_argument(
        '--audio_file', '-a',
        type=str,
        required=True,
        help='播客音频文件路径'
    )
    
    parser.add_argument(
        '--whisper_model_path', '-w',
        type=str,
        required=True,
        help='Whisper模型的本地路径或模型名称'
    )
    
    parser.add_argument(
        '--hf_token', '-t',
        type=str,
        required=True,
        help='Hugging Face API令牌'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='.',
        help='输出目录（默认为当前目录）'
    )
    
    parser.add_argument(
        '--diarization_model', '-d',
        type=str,
        default='pyannote/speaker-diarization-3.1',
        help='说话人分割模型名称'
    )
    
    parser.add_argument(
        '--summarization_model', '-s',
        type=str,
        default='facebook/bart-large-cnn',
        help='摘要模型名称或路径'
    )
    
    parser.add_argument(
        '--transcriber_type',
        type=str,
        choices=['openai-whisper', 'whisper-cpp'],
        default='openai-whisper',
        help='选择转录器类型：原生OpenAI Whisper或whisper.cpp'
    )
    
    parser.add_argument(
        '--summarizer_type',
        type=str,
        choices=['transformers', 'qwen'],
        default='transformers',
        help='选择摘要器类型：Hugging Face Transformers或Qwen本地模型'
    )
    
    parser.add_argument(
        '--summarizer_device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='摘要模型运行设备（默认为CUDA）'
    )
    
    parser.add_argument(
        '--no_4bit_quantization',
        action='store_true',
        help='禁用4bit量化加载Qwen模型（仅对Qwen模型有效）'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='生成摘要的温度参数（仅对Qwen模型有效）'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='生成摘要的top_p参数（仅对Qwen模型有效）'
    )
    
    parser.add_argument(
        '--max_summary_length',
        type=int,
        default=150,
        help='最大摘要长度'
    )
    
    parser.add_argument(
        '--min_summary_length',
        type=int,
        default=30,
        help='最小摘要长度'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志'
    )
    
    return parser.parse_args()


def main():
    """命令行入口点"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(args.verbose)
    
    try:
        # 验证路径
        if not os.path.isfile(args.audio_file):
            logger.error(f"音频文件不存在: {args.audio_file}")
            sys.exit(1)
            
        # 对于whisper-cpp，模型可以是模型名称也可以是模型文件路径
        if args.transcriber_type == 'openai-whisper':
            if not os.path.isfile(args.whisper_model_path):
                logger.error(f"Whisper模型文件不存在: {args.whisper_model_path}")
                sys.exit(1)
        
        # 确保输出目录存在
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        audio_basename = os.path.basename(args.audio_file)
        audio_name = os.path.splitext(audio_basename)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        transcript_file = output_dir / f"{audio_name}_{timestamp}_transcript.txt"
        summary_file = output_dir / f"{audio_name}_{timestamp}_summary.txt"
        
        logger.info(f"初始化播客转录器，使用转录器类型: {args.transcriber_type}，摘要器类型: {args.summarizer_type}...")
        transcriber = PodcastTranscriber(
            whisper_model_path=args.whisper_model_path,
            hf_token=args.hf_token,
            diarization_model=args.diarization_model,
            summarization_model=args.summarization_model,
            transcriber_type=args.transcriber_type,
            summarizer_type=args.summarizer_type,
            summarizer_device=args.summarizer_device,
            summarizer_load_in_4bit=not args.no_4bit_quantization
        )
        
        # 准备摘要额外参数
        summary_kwargs = {}
        
        # 对于Qwen模型，添加相关参数
        if args.summarizer_type == 'qwen':
            summary_kwargs.update({
                'temperature': args.temperature,
                'top_p': args.top_p
            })
        
        logger.info("开始转录和摘要...")
        summary, transcript = transcriber.transcribe_and_summarize(
            audio_file=args.audio_file,
            max_summary_length=args.max_summary_length,
            min_summary_length=args.min_summary_length,
            output_transcript_file=transcript_file,
            output_summary_file=summary_file,
            **summary_kwargs
        )
        
        logger.info(f"转录文本已保存到: {transcript_file}")
        logger.info(f"摘要文本已保存到: {summary_file}")
        
        # 打印摘要
        print("\n===== 播客摘要 =====")
        print(summary)
        print("====================\n")
        
    except Exception as e:
        logger.exception(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)
    
    
if __name__ == "__main__":
    main() 