"""
src/logging_config.py
════════════════════════════════════════════════════════════════════════════
Centralized logging setup for the entire project.

Features:
  - Mỗi lần chạy tạo thư mục artifacts/logs/train_<YYYYMMDD_HHMMSS>/
  - RotatingFileHandler chia nhỏ (max 5MB/file), KHÔNG xóa file cũ
  - Per-component level control qua configs/logging.yaml
  - Console mặc định WARNING (ít output trong notebook)

Usage (call once at entry point):
    from src.logging_config import setup_logging
    setup_logging()                    # Đọc từ configs/logging.yaml
    setup_logging(config_path="...")   # Custom config path
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────
# Per-logger filter: cho phép set level riêng cho từng handler
# ─────────────────────────────────────────────────────────────────────────

class _LevelFilter(logging.Filter):
    """Filter records below a given level for a specific handler."""

    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.level


# ─────────────────────────────────────────────────────────────────────────
# Main setup function
# ─────────────────────────────────────────────────────────────────────────

# Track whether setup has been called (avoid duplicate handlers)
_setup_done = False
_current_log_dir: Path | None = None


def get_log_dir() -> Path | None:
    """Return the current run's log directory (None if setup not called)."""
    return _current_log_dir


def setup_logging(config_path: str = "configs/logging.yaml") -> Path:
    """
    Configure logging for the entire project.

    Args:
        config_path: Path to logging YAML config.

    Returns:
        Path to the created log directory for this run.
    """
    global _setup_done, _current_log_dir

    # Idempotent — chỉ setup 1 lần
    if _setup_done and _current_log_dir is not None:
        return _current_log_dir

    # ── Load config ───────────────────────────────────────────────
    cfg = _load_config(config_path)
    file_cfg = cfg.get("file", {})
    console_cfg = cfg.get("console", {})
    loggers_cfg = cfg.get("loggers", {})

    file_level = _parse_level(file_cfg.get("level", "DEBUG"))
    max_bytes = int(file_cfg.get("max_bytes", 5_000_000))
    backup_count = int(file_cfg.get("backup_count", 100))
    console_level = _parse_level(console_cfg.get("level", "WARNING"))

    # ── Create run directory ──────────────────────────────────────
    base_dir = Path(cfg.get("base_dir", "artifacts/logs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base_dir / f"train_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    _current_log_dir = log_dir

    # ── Root logger ───────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Allow all; handlers filter
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(console_level)
    console_h.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname).1s %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console_h)

    # ── File handler (RotatingFileHandler) ─────────────────────────
    log_file = log_dir / "train.log"
    file_h = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_h.setLevel(file_level)
    file_h.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_h)

    # ── Per-component level overrides ─────────────────────────────
    for logger_name, levels in loggers_cfg.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  # Let handlers decide

        # Per-logger file level
        logger_file_level = _parse_level(levels.get("file", "DEBUG"))
        logger_console_level = _parse_level(levels.get("console", "WARNING"))

        # Set logger level to min of both targets
        effective = min(logger_file_level, logger_console_level)
        logger.setLevel(effective)

    # ── Suppress noisy third-party loggers ────────────────────────
    for noisy in ("httpx", "httpcore", "gradio_client", "urllib3",
                  "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _setup_done = True

    # Log startup info (goes to file, not console since it's DEBUG)
    logging.getLogger(__name__).debug(
        f"Logging initialized: dir={log_dir}, "
        f"console={logging.getLevelName(console_level)}, "
        f"file={logging.getLevelName(file_level)}, "
        f"max_bytes={max_bytes}, backup_count={backup_count}"
    )

    return log_dir


def _load_config(path: str) -> dict:
    """Load YAML config, return empty dict if not found."""
    p = Path(path)
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def _parse_level(level_str: str) -> int:
    """Convert level string to logging constant."""
    return getattr(logging, level_str.upper(), logging.DEBUG)
