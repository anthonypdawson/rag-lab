import logging
from .config import Config
import click

class ClickColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'magenta',
    }

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelname, None)
        if color:
            return click.style(msg, fg=color)
        return msg

def get_logger(name=None):
    """Get a logger configured with loglevel from config and colorized output."""
    config = Config()
    loglevel = config.get('logging.level', 'info').upper()
    logger = logging.getLogger(name if name else __name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = ClickColorFormatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, loglevel, logging.INFO))
    return logger
