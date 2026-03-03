"""Shared colored log formatter for scripts."""

import logging


class ColorFormatter(logging.Formatter):
    GREY = "\033[90m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"

    def format(self, record):
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname
        if level == "WARNING":
            lvl_color = self.YELLOW
        elif level == "ERROR":
            lvl_color = self.RED
        else:
            lvl_color = self.GREEN
        msg = record.getMessage()
        if msg.startswith("["):
            bracket_end = msg.find("]") + 1
            msg = f"{self.CYAN}{msg[:bracket_end]}{self.RESET} {msg[bracket_end:].strip()}"
        return (
            f"{self.GREY}{ts}{self.RESET} "
            f"{lvl_color}{level}{self.RESET} "
            f"{msg}"
        )


def setup():
    """Configure root logger with colored output."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)
