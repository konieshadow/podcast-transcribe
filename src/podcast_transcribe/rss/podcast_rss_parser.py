import requests
import feedparser
# from dataclasses import dataclass, field # 已移除
from typing import Optional # , List, Dict # List 和 Dict 不再需要
from datetime import datetime
import time

from ..schemas import PodcastEpisode, PodcastChannel

def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        # feedparser 已经将日期解析为 time.struct_time 类型
        # 我们将其转换为 datetime 类型
        if isinstance(date_str, time.struct_time):
            return datetime.fromtimestamp(time.mktime(date_str))
        # 如果 feedparser 解析失败或返回字符串，则回退使用其他字符串格式解析
        # 这是一种常见的 RSS 日期格式
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    except (ValueError, TypeError):
        try:
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z') # 处理 GMT, EST 等时区
        except (ValueError, TypeError):
            # 如果时区缺失或无法解析，则尝试不带时区解析
            try:
                return datetime.strptime(date_str[:-6], '%a, %d %b %Y %H:%M:%S')
            except (ValueError, TypeError):
                print(f"Warning: Could not parse date string: {date_str}")
                return None

def fetch_rss_content(rss_url: str) -> Optional[bytes]:
    """
    通过 HTTP 请求获取 RSS feed 的内容。

    参数:
        rss_url: 播客 RSS feed 的 URL。

    返回:
        bytes 类型的 RSS 内容，如果获取失败则返回 None。
    """
    try:
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status() # 针对 HTTP 错误抛出异常
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"获取 RSS feed 时出错: {e}")
        return None

def parse_rss_xml_content(rss_content: bytes) -> Optional[PodcastChannel]:
    """
    解析播客 RSS XML 内容，并返回其主要信息和剧集详情。

    参数:
        rss_content: bytes 类型的 RSS XML 内容。

    返回:
        一个包含已解析信息的 PodcastChannel 对象，如果解析失败则返回 None。
    """
    feed = feedparser.parse(rss_content)

    if feed.bozo:
        # 如果 feed 格式不正确，bozo 为 True
        # feed.bozo_exception 包含异常信息
        print(f"警告: RSS feed 可能格式不正确。Bozo 异常: {feed.bozo_exception}")
        # 即使格式不完全正确，feedparser 通常仍会尝试解析，所以我们不在此处直接返回 None
        # 但如果关键的 feed 或 channel_info 缺失，后续会自然失败

    channel_info = feed.get('feed', {})
    if not channel_info: # 如果连基本的 feed 结构都没有，则认为解析失败
        print("错误: RSS 内容无法解析为有效的 feed 结构。")
        return None

    podcast_channel = PodcastChannel(
        title=channel_info.get('title'),
        link=channel_info.get('link'),
        description=channel_info.get('subtitle') or channel_info.get('description'),
        language=channel_info.get('language'),
        image_url=channel_info.get('image', {}).get('href') if channel_info.get('image') else None,
        author=channel_info.get('author') or channel_info.get('itunes_author'),
        last_build_date=_parse_date(channel_info.get('updated_parsed') or channel_info.get('published_parsed'))
    )

    for entry in feed.entries:
        # 确定 shownotes：优先使用 content:encoded，然后是 itunes:summary，其次是 description/summary
        shownotes = None
        if 'content' in entry: # 检查 <content:encoded>
            # feedparser 将多个内容版本存储在一个列表中
            # 通常第一个是最详细的
            shownotes = entry.content[0].value if entry.content and entry.content[0].value else None
        
        if not shownotes and 'itunes_summary' in entry:
            shownotes = entry.itunes_summary
        
        if not shownotes: # 回退到 summary 或 description
            shownotes = entry.get('summary') or entry.get('description')

        # 从 enclosures 获取音频 URL
        audio_url = None
        if 'enclosures' in entry:
            for enc in entry.enclosures:
                if enc.get('type', '').startswith('audio/'):
                    audio_url = enc.get('href')
                    break
        
        # 解析特定于剧集的 iTunes 标签
        itunes_season = None
        try:
            itunes_season_str = entry.get('itunes_season')
            if itunes_season_str:
                itunes_season = int(itunes_season_str)
        except (ValueError, TypeError):
            pass # 如果不是有效整数则忽略

        itunes_episode_number = None
        try:
            itunes_episode_number_str = entry.get('itunes_episode')
            if itunes_episode_number_str:
                itunes_episode_number = int(itunes_episode_number_str)
        except (ValueError, TypeError):
            pass # 如果不是有效整数则忽略

        episode = PodcastEpisode(
            title=entry.get('title'),
            link=entry.get('link'),
            published_date=_parse_date(entry.get('published_parsed')),
            summary=entry.get('summary'), # 这通常是较短的版本
            shownotes=shownotes, # 这是我们尝试获取的更详细版本
            audio_url=audio_url,
            guid=entry.get('id') or entry.get('guid'),
            duration=entry.get('itunes_duration'),
            episode_type=entry.get('itunes_episodetype'),
            season=itunes_season,
            episode_number=itunes_episode_number
        )
        podcast_channel.episodes.append(episode)

    return podcast_channel

def get_podcast_data_from_url(rss_url: str) -> Optional[PodcastChannel]:
    """
    从给定的 RSS URL 获取并解析播客数据。

    参数:
        rss_url: 播客 RSS feed 的 URL。

    返回:
        一个包含已解析信息的 PodcastChannel 对象，如果获取或解析失败则返回 None。
    """
    rss_content = fetch_rss_content(rss_url)
    if rss_content:
        return parse_rss_xml_content(rss_content)
    return None

# 为了保持与之前调用方式的兼容性（如果其他地方已经在使用 parse_podcast_rss），
# 可以将原来的 parse_podcast_rss 指向新的顶层函数。
# 或者，如果确定没有其他地方使用旧名称，可以直接删除旧的 parse_podcast_rss。
# 这里我们选择重命名/替换的方式。
parse_podcast_rss = get_podcast_data_from_url