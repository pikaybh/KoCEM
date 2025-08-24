import logging, os
from datetime import datetime

from dotenv import load_dotenv


load_dotenv()


def set_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 파일 핸들러 (DEBUG 이상 기록)
    logger_dir = os.path.join(os.getenv("LOG_DIR", "logs"), datetime.now().strftime('%Y%m%d'))
    os.makedirs(logger_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(logger_dir, f"{name}.log"), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 (INFO 이상만 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 루트 로거 핸들러 제거 (중복 방지)
    logging.getLogger().handlers.clear()

    return logger


__all__ = ['set_logger']