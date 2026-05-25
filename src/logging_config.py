"""
src/logging_config.py
════════════════════════════════════════════════════════════════════════════
Centralized logging setup for the entire project.

Usage (call once at entry point):
    from src.logging_config import setup_logging
    setup_logging()                    # INFO to console, DEBUG to file
    setup_logging(level="DEBUG")       # DEBUG everywhere (verbose)
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import logging
import os
import sys


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/train.log",
    file_level: str = "DEBUG",
) -> None:
    """
    Configure logging for the entire project.

    Args:
        level: Console log level (INFO, DEBUG, WARNING, etc.)
        log_file: Path to log file (DEBUG level always)
        file_level: File handler level
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    console_level = getattr(logging, level.upper(), logging.INFO)
    f_level = getattr(logging, file_level.upper(), logging.DEBUG)

    # Root logger — catches everything
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Allow all; handlers filter

    # Clear existing handlers (avoid duplicates on re-call)
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname).1s %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # ── File handler (ghi hết DEBUG) ──────────────────────────────
    file_h = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_h.setLevel(f_level)
    file_h.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_h)

    # ── Suppress noisy third-party loggers ────────────────────────
    for noisy in ("httpx", "httpcore", "gradio_client", "urllib3",
                  "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
