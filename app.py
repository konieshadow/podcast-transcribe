#!/usr/bin/env python3
"""
播客转录工具 - 主启动文件
这个文件用于启动 Gradio WebUI 应用
"""

import sys
import os

# 将 src 目录添加到 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
    """主函数：启动 WebUI 应用"""
    try:
        # 导入并启动 webui 应用
        from podcast_transcribe.webui.app import demo
        
        print("🎙️ 启动播客转录工具...")
        print("📍 WebUI 将在浏览器中打开")
        print("🔗 默认地址: http://localhost:7860")
        print("⏹️  按 Ctrl+C 停止服务")
        
        # 启动 Gradio 应用
        demo.launch(
            debug=True,
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7860,       # 指定端口
            share=False,            # 不生成公开链接
            inbrowser=True          # 自动在浏览器中打开
        )
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
