from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from datetime import datetime

@dataclass
class EnhancedSegment:
    """增强的转录分段，包含说话人信息"""
    start: float  # 开始时间（秒）
    end: float  # 结束时间（秒）
    text: str  # 转录的文本
    speaker: str  # 说话人ID
    language: str  # 检测到的语言
    speaker_name: Optional[str] = None  # 识别出的说话人名称


@dataclass
class CombinedTranscriptionResult:
    """结合ASR和说话人分离的转录结果"""
    segments: List[EnhancedSegment]  # 包含说话人和文本的分段
    text: str  # 完整转录文本
    language: str  # 检测到的语言
    num_speakers: int  # 检测到的说话人数量 


@dataclass
class PodcastEpisode:
    title: Optional[str] = None
    link: Optional[str] = None
    published_date: Optional[datetime] = None
    summary: Optional[str] = None # 简短摘要
    shownotes: Optional[str] = None # 详细的shownotes，通常是HTML格式
    audio_url: Optional[str] = None
    guid: Optional[str] = None
    duration: Optional[str] = None # 例如，来自 <itunes:duration>
    episode_type: Optional[str] = None # 例如，来自 <itunes:episodetype>
    season: Optional[int] = None # 例如，来自 <itunes:season>
    episode_number: Optional[int] = None # 例如，来自 <itunes:episode>

@dataclass
class PodcastChannel:
    title: Optional[str] = None
    link: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    image_url: Optional[str] = None
    author: Optional[str] = None # 例如，来自 <itunes:author>
    last_build_date: Optional[datetime] = None
    episodes: List[PodcastEpisode] = field(default_factory=list) 


@dataclass
class TranscriptionResult:
    """转录结果数据类"""
    text: str  # 转录的文本
    segments: List[Dict[str, Union[float, str]]]  # 包含时间戳的分段
    language: str  # 检测到的语言


@dataclass
class DiarizationResult:
    """说话人分离结果数据类"""
    segments: List[Dict[str, Union[float, str, int]]]  # 包含时间戳和说话人ID的分段
    num_speakers: int  # 检测到的说话人数量 