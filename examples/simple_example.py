#!/usr/bin/env python3
"""
播客转录和摘要的简单示例
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.podcast_transcribe.transcriber import PodcastTranscriber


def main():
    """简单示例的主函数"""
    
    # 需要设置这些参数
    audio_file = Path.joinpath(Path(__file__).parent, "input", "lex_ai_john_carmack_1.wav")  # 播客音频文件路径
    whisper_model_path = "large-v3-turbo"  # Whisper模型路径
    hf_token = "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"  # Hugging Face API 令牌
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 生成输出文件路径
    audio_name = Path(audio_file).stem
    transcript_file = output_dir / f"{audio_name}_transcript.txt"
    summary_file = output_dir / f"{audio_name}_summary.txt"
    
    # 初始化转录器
    transcriber = PodcastTranscriber(
        whisper_model_path=whisper_model_path,
        hf_token=hf_token
    )
    
    # 执行转录和摘要
    summary, transcript = transcriber.transcribe_and_summarize(
        audio_file=audio_file,
        output_transcript_file=transcript_file,
        output_summary_file=summary_file
    )
    
    # 打印摘要
    print("\n===== 播客摘要 =====")
    print(summary)
    print("====================\n")
    
    print(f"转录文本已保存到: {transcript_file}")
    print(f"摘要文本已保存到: {summary_file}")


if __name__ == "__main__":
    main() 