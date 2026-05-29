from __future__ import annotations

import logging

LOGGER_NAME = "deep_agents"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the deep_agents namespace."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure console logging for examples and local runs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
