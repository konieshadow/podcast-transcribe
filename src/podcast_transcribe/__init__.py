import logging

# 设置根日志级别为INFO，这样第三方包默认使用INFO级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 单独设置podcast_transcribe包的日志级别
package_logger = logging.getLogger("podcast_transcribe")
package_logger.setLevel(logging.INFO)