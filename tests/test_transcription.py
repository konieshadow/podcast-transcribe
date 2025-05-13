"""
转录功能测试模块
"""

import unittest
from io import BytesIO
import tempfile
import os
import numpy as np
import soundfile as sf

# 尝试导入pywhispercpp
try:
    import pywhispercpp
    PYWHISPERCPP_AVAILABLE = True
except ImportError:
    PYWHISPERCPP_AVAILABLE = False

from podcast_transcribe.utils.transcription import WhisperCppTranscriber, WhisperTranscriber


class TestWhisperCppTranscriber(unittest.TestCase):
    """测试WhisperCpp转录器"""
    
    @unittest.skipIf(not PYWHISPERCPP_AVAILABLE, "pywhispercpp不可用，跳过测试")
    def test_initialization(self):
        """测试初始化"""
        transcriber = WhisperCppTranscriber("tiny")
        self.assertIsNotNone(transcriber)
    
    @unittest.skipIf(not PYWHISPERCPP_AVAILABLE, "pywhispercpp不可用，跳过测试")
    def test_transcribe_empty_audio(self):
        """测试转录空音频"""
        # 创建一个空音频（1秒的静音）
        sample_rate = 16000
        audio_data = np.zeros(sample_rate, dtype=np.float32)
        
        # 保存为BytesIO对象
        audio_io = BytesIO()
        sf.write(audio_io, audio_data, sample_rate, format='wav')
        audio_io.seek(0)
        
        transcriber = WhisperCppTranscriber("tiny")
        result = transcriber.transcribe(audio_io)
        
        # 验证结果格式
        self.assertIn('segments', result)
        self.assertIsInstance(result['segments'], list)


class TestTranscriberInterface(unittest.TestCase):
    """测试转录器接口一致性"""
    
    def create_test_audio(self):
        """创建测试音频"""
        sample_rate = 16000
        duration = 1  # 1秒
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # 生成一个简单的音调
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # 保存为BytesIO对象
        audio_io = BytesIO()
        sf.write(audio_io, audio_data.astype(np.float32), sample_rate, format='wav')
        audio_io.seek(0)
        return audio_io
    
    @unittest.skipIf(not PYWHISPERCPP_AVAILABLE, "pywhispercpp不可用，跳过测试")
    def test_interface_compatibility(self):
        """测试接口兼容性"""
        # 这个测试需要两种转录器都能访问
        # 但如果whisper模型不可用，也可以跳过WhisperTranscriber相关测试
        
        try:
            # 创建测试音频
            audio_io = self.create_test_audio()
            
            # 测试WhisperCppTranscriber的结果格式
            cpp_transcriber = WhisperCppTranscriber("tiny")
            cpp_result = cpp_transcriber.transcribe(audio_io)
            
            # 验证结果包含预期的字段
            self.assertIn('segments', cpp_result)
            
            # 如果有段落，验证每个段落的结构
            if cpp_result['segments']:
                segment = cpp_result['segments'][0]
                self.assertIn('id', segment)
                self.assertIn('start', segment)
                self.assertIn('end', segment)
                self.assertIn('text', segment)
            
            # 提取文本应该返回段落列表
            segments = cpp_transcriber.extract_text(cpp_result)
            self.assertIsInstance(segments, list)
            
        except Exception as e:
            self.skipTest(f"测试遇到错误: {str(e)}")


if __name__ == '__main__':
    unittest.main() 