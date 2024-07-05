from pathlib import Path
import logging
from logging import handlers


class LoggerFactory:
    def __init__(self, filename: str):
        file_handler = handlers.RotatingFileHandler(
            filename=Path(__file__).with_name(f"{filename}.log"),
            mode="a",
            maxBytes=1 * 1024 * 1024,
            backupCount=0,
            encoding=None,
            delay=False,
        )
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            handlers=[file_handler],
        )

    def create(self, name: str) -> logging.Logger:
        return logging.getLogger(name)
