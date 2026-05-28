from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_env(path: str | Path = ".env", *, override: bool = False) -> bool:
    """Load environment variables from a dotenv file.

    Values already present in the process environment are preserved by default.
    Returns whether a dotenv file was found and loaded.
    """
    return load_dotenv(dotenv_path=path, override=override)
