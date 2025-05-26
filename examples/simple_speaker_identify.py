# 添加项目根目录到Python路径
import json
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.podcast_transcribe.rss.podcast_rss_parser import parse_rss_xml_content
from src.podcast_transcribe.schemas import EnhancedSegment, CombinedTranscriptionResult
from src.podcast_transcribe.summary.speaker_identify import SpeakerIdentifier

if __name__ == '__main__':
    transcribe_result_dump_file = Path.joinpath(Path(__file__).parent, "output", "lex_ai_john_carmack_1.transcription.json")
    podcast_rss_xml_file = Path.joinpath(Path(__file__).parent, "input", "lexfridman.com.rss.xml")

    # Load the transcription result
    if not os.path.exists(transcribe_result_dump_file):
        print(f"错误：转录结果文件 '{transcribe_result_dump_file}' 不存在。请先运行 combined_transcription.py 生成结果。")
        sys.exit(1)

    with open(transcribe_result_dump_file, "r", encoding="utf-8") as f:
        # transcription_result = json.load(f) # 旧代码
        data = json.load(f)
        segments_data = data.get("segments", [])
        # 确保 segments_data 中的每个元素都是字典，以避免在 EnhancedSegment(**seg) 时出错
        # 假设 EnhancedSegment 的字段与 JSON 中 segment 字典的键完全对应
        enhanced_segments = []
        for seg_dict in segments_data:
            if isinstance(seg_dict, dict):
                enhanced_segments.append(EnhancedSegment(**seg_dict))
            else:
                # 处理非字典类型 segment 的情况，例如记录日志或抛出错误
                print(f"警告: 在JSON中发现非字典类型的segment: {seg_dict}")
        
        transcription_result = CombinedTranscriptionResult(
            segments=enhanced_segments,
            text=data.get("text", ""),
            language=data.get("language", ""),
            num_speakers=data.get("num_speakers", 0)
        )

    # 打印加载的 CombinedTranscriptionResult 对象的一些信息以供验证
    print(f"\\n成功从JSON加载 CombinedTranscriptionResult 对象:")
    print(f"类型: {type(transcription_result)}")

    # Load the podcast RSS XML file
    with open(podcast_rss_xml_file, "r") as f:
        podcast_rss_xml = f.read()
        mock_podcast_info = parse_rss_xml_content(podcast_rss_xml)

    
    # 查找标题已 "#309" 开头的剧集
    mock_episode_info = next((episode for episode in mock_podcast_info.episodes if episode.title.startswith("#309")), None)
    if not mock_episode_info:
        raise ValueError("Could not find episode with title starting with '#309'")
    

    speaker_identifier = SpeakerIdentifier(
        llm_model_name="mlx-community/gemma-3-12b-it-4bit-DWQ",
        llm_provider="gemma-mlx"
    )

    # 3. Call the function
    print("\\n--- Test Case 1: Normal execution ---")
    speaker_names = speaker_identifier.recognize_speaker_names(transcription_result.segments, mock_podcast_info, mock_episode_info)
    print("\\nRecognized Speaker Names (Test Case 1):")
    print(json.dumps(speaker_names, ensure_ascii=False, indent=2))