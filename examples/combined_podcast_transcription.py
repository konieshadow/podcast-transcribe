# 添加项目根目录到Python路径
import json
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.podcast_transcribe.transcriber import transcribe_podcast_audio
from src.podcast_transcribe.audio import load_audio
from src.podcast_transcribe.rss.podcast_rss_parser import parse_rss_xml_content
from podcast_transcribe.llm.llm_gemma_mlx import GemmaMLXChatCompletion
from src.podcast_transcribe.schemas import EnhancedSegment, CombinedTranscriptionResult
from src.podcast_transcribe.summary.speaker_identify import recognize_speaker_names

def main():
    """主函数"""
    podcast_rss_xml_file = Path.joinpath(Path(__file__).parent, "input", "lexfridman.com.rss.xml")
    audio_file = Path.joinpath(Path(__file__).parent, "input", "lex_ai_john_carmack_1.wav")  # 播客音频文件路径
    # audio_file = Path("/Users/konie/Desktop/voices/lex_ai_john_carmack_30.wav")
    
    # 模型配置
    asr_model_name = "mlx-community/"  # ASR模型名称
    diarization_model_name = "pyannote/speaker-diarization-3.1"  # 说话人分离模型名称
    llm_model_path = "mlx-community/gemma-3-12b-it-4bit-DWQ"
    device = "mps"  # 设备类型
    segmentation_batch_size = 64
    parallel = True

    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误：文件 '{audio_file}' 不存在")
        return 1
    
    try:
        print(f"正在加载音频文件: {audio_file}")
        # 加载音频文件
        audio, _ = load_audio(audio_file)
        
        print(f"音频信息: 时长={audio.duration_seconds:.2f}秒, 通道数={audio.channels}, 采样率={audio.frame_rate}Hz")


    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Load the podcast RSS XML file
    with open(podcast_rss_xml_file, "r") as f:
        podcast_rss_xml = f.read()
        mock_podcast_info = parse_rss_xml_content(podcast_rss_xml)

    
    # 查找标题已 "#309" 开头的剧集
    mock_episode_info = next((episode for episode in mock_podcast_info.episodes if episode.title.startswith("#309")), None)
    if not mock_episode_info:
        raise ValueError("Could not find episode with title starting with '#309'")

    result = transcribe_podcast_audio(audio,
                             podcast_info=mock_podcast_info,
                             episode_info=mock_episode_info,
                             asr_model_name=asr_model_name,
                             diarization_model_name=diarization_model_name,
                             llm_model_name=llm_model_path,
                             device=device,
                             segmentation_batch_size=segmentation_batch_size,
                             parallel=parallel,
                             llm_model_name=llm_model_path)
    
    # 输出结果
    print("\n转录结果:")
    print("-" * 50)
    print(f"检测到的语言: {result.language}")
    print(f"检测到的说话人数量: {result.num_speakers}")
    print(f"总文本长度: {len(result.text)} 字符")
    
    # 输出每个说话人的部分
    speakers = set(segment.speaker for segment in result.segments)
    for speaker in sorted(speakers):
        speaker_segments = [seg for seg in result.segments if seg.speaker == speaker]
        total_duration = sum(seg.end - seg.start for seg in speaker_segments)
        print(f"\n说话人 {speaker}: 共 {len(speaker_segments)} 个片段, 总时长 {total_duration:.2f} 秒")
    
    # 输出详细分段信息
    print("\n详细分段信息:")
    for i, segment in enumerate(result.segments, 1):
        if i <= 20 or i > len(result.segments) - 20:  # 仅显示前20个和后20个分段
            print(f"段落 {i}/{len(result.segments)}: [{segment.start:.2f}s - {segment.end:.2f}s] 说话人: {segment.speaker_name if segment.speaker_name else segment.speaker} 文本: {segment.text}")
        elif i == 21:
            print("... 省略中间部分 ...")

if __name__ == '__main__':
    main()