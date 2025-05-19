
# 添加项目根目录到Python路径
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.podcast_transcribe.rss.podcast_rss_parser import parse_podcast_rss

if __name__ == '__main__':
    # 使用示例:
    lex_fridman_rss = "https://lexfridman.com/feed/podcast/"
    print(f"正在解析 Lex Fridman Podcast RSS: {lex_fridman_rss}")
    podcast_data = parse_podcast_rss(lex_fridman_rss)

    if podcast_data:
        print(f"Podcast Title: {podcast_data.title}")
        print(f"Podcast Link: {podcast_data.link}")
        print(f"Podcast Description: {podcast_data.description[:200] if podcast_data.description else 'N/A'}...")
        print(f"Podcast Author: {podcast_data.author}")
        print(f"Podcast Image URL: {podcast_data.image_url}")
        print(f"Total episodes found: {len(podcast_data.episodes)}")

        if podcast_data.episodes:
            print("\n--- Sample Episode ---")
            sample_episode = podcast_data.episodes[0]
            print(f"  标题: {sample_episode.title}")
            print(f"  发布日期: {sample_episode.published_date}")
            print(f"  链接: {sample_episode.link}")
            print(f"  音频 URL: {sample_episode.audio_url}")
            print(f"  GUID: {sample_episode.guid}")
            print(f"  时长: {sample_episode.duration}")
            print(f"  季: {sample_episode.season}")
            print(f"  集数: {sample_episode.episode_number}")
            print(f"  剧集类型: {sample_episode.episode_type}")
            print(f"  摘要: {sample_episode.summary[:200] if sample_episode.summary else 'N/A'}...")
            print(f"  Shownotes (前 300 字符): {sample_episode.shownotes[:300] if sample_episode.shownotes else 'N/A'}...")
    else:
        print("解析播客 RSS feed 失败。")
