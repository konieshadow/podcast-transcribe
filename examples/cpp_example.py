#!/usr/bin/env python3
"""
使用whisper.cpp进行播客转录和摘要的示例
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pywhispercpp
except ImportError:
    print("错误: 需要安装pywhispercpp")
    print("请使用: pip install pywhispercpp")
    sys.exit(1)

from src.podcast_transcribe.transcriber import PodcastTranscriber


def main():
    """whisper.cpp示例的主函数"""
    
    # 需要设置这些参数
    audio_file = Path.joinpath(Path(__file__).parent, "input", "lex_ai_john_carmack_1.wav")  # 播客音频文件路径
    whisper_model_path = "large-v3-turbo"  # 模型名称或路径，将自动下载
    hf_token = "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"  # Hugging Face API 令牌
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 生成输出文件路径
    audio_name = Path(audio_file).stem
    transcript_file = output_dir / f"{audio_name}_cpp_transcript.txt"
    summary_file = output_dir / f"{audio_name}_cpp_summary.txt"
    
    # 初始化转录器，使用whisper.cpp引擎
    transcriber = PodcastTranscriber(
        whisper_model_path=whisper_model_path,
        hf_token=hf_token,
        transcriber_type="whisper-cpp"  # 指定使用whisper.cpp
    )
    
    # 执行转录和摘要
    summary, transcript = transcriber.transcribe_and_summarize(
        audio_file=audio_file,
        output_transcript_file=transcript_file,
        output_summary_file=summary_file
    )
    
    # 打印摘要
    print("\n===== 播客摘要 (使用whisper.cpp) =====")
    print(summary)
    print("=====================================\n")
    
    print(f"转录文本已保存到: {transcript_file}")
    print(f"摘要文本已保存到: {summary_file}")


if __name__ == "__main__":
    main() 