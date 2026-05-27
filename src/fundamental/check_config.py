"""
src/fundamental/check_config.py
════════════════════════════════════════════════════════════════════════════
Kiểm tra cấu hình src/fundamental/config.yaml.
Nếu các field quan trọng trống → prompt người dùng nhập tay.

Usage:
    python -m src.fundamental.check_config
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv, set_key

log = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
_CONFIG_PATH = _PACKAGE_DIR / "config.yaml"
_ENV_PATH = Path(".env")  # Project root .env


def _load_config() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _prompt_value(field_name: str, current: str | None, secret: bool = False) -> str:
    """Hiển thị input cho user nhập giá trị."""
    display = "(trống)" if not current else current
    if secret and current:
        display = current[:8] + "..." + current[-4:]

    log.info(f"📋 {field_name}")
    log.info(f"   Hiện tại: {display}")

    if secret:
        from getpass import getpass
        value = getpass(f"   Nhập giá trị mới (Enter để giữ nguyên): ")
    else:
        value = input(f"   Nhập giá trị mới (Enter để giữ nguyên): ").strip()

    return value if value else (current or "")


def check_and_fix():
    """Kiểm tra config và prompt user nhập nếu thiếu."""
    log.info("=" * 60)
    log.info("🔍 Kiểm tra cấu hình Analysis Pipeline")
    log.info("=" * 60)

    cfg = _load_config()
    changed = False

    # Load .env
    load_dotenv(_ENV_PATH)

    # ── 1. Check llm_backend ─────────────────────────────────────
    backend = cfg.get("llm_backend", "")
    if not backend:
        log.warning("llm_backend chưa được set!")
        backend = input("   Chọn backend LLM [ckey/gradio] (mặc định: ckey): ").strip() or "ckey"
        cfg["llm_backend"] = backend
        changed = True
    else:
        log.info(f"✓ llm_backend: {backend}")

    # ── 2. Check CKey config ─────────────────────────────────────
    ckey_cfg = cfg.get("ckey", {})
    if not ckey_cfg:
        ckey_cfg = {}
        cfg["ckey"] = ckey_cfg

    ckey_model = ckey_cfg.get("model", "")
    if not ckey_model:
        ckey_model = _prompt_value("ckey.model (tên model CKey)", ckey_model)
        ckey_cfg["model"] = ckey_model
        changed = True
    else:
        log.info(f"✓ ckey.model: {ckey_model}")

    # Check CKEY_API in .env
    ckey_api = os.environ.get("CKEY_API", "")
    if not ckey_api:
        log.warning("CKEY_API chưa có trong .env!")
        ckey_api = _prompt_value("CKEY_API (API key cho CKey)", "", secret=True)
        if ckey_api:
            _ENV_PATH.touch(exist_ok=True)
            set_key(str(_ENV_PATH), "CKEY_API", ckey_api)
            os.environ["CKEY_API"] = ckey_api
            log.info("✓ Đã lưu CKEY_API vào .env")
    else:
        log.info(f"✓ CKEY_API: {ckey_api[:8]}...{ckey_api[-4:]}")

    # ── 3. Check LLM Gradio config ──────────────────────────────
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg:
        llm_cfg = {}
        cfg["llm"] = llm_cfg

    llm_model = llm_cfg.get("model", "")
    if not llm_model:
        llm_model = _prompt_value("llm.model (tên model Gradio LLM)", llm_model)
        llm_cfg["model"] = llm_model
        changed = True
    else:
        log.info(f"✓ llm.model: {llm_model}")

    llm_url = llm_cfg.get("gradio_url", "")
    if not llm_url:
        llm_url = _prompt_value("llm.gradio_url (URL Gradio LLM)", llm_url)
        llm_cfg["gradio_url"] = llm_url
        changed = True
    else:
        log.info(f"✓ llm.gradio_url: {llm_url}")

    # ── 4. Check Embedding Gradio config ─────────────────────────
    embed_cfg = cfg.get("embedding", {})
    if not embed_cfg:
        embed_cfg = {}
        cfg["embedding"] = embed_cfg

    embed_url = embed_cfg.get("gradio_url", "")
    if not embed_url:
        embed_url = _prompt_value("embedding.gradio_url (URL Gradio Embedding)", embed_url)
        embed_cfg["gradio_url"] = embed_url
        changed = True
    else:
        log.info(f"✓ embedding.gradio_url: {embed_url}")

    embed_model = embed_cfg.get("model", "")
    if not embed_model:
        embed_model = _prompt_value("embedding.model (tên model Embedding)", embed_model)
        embed_cfg["model"] = embed_model
        changed = True
    else:
        log.info(f"✓ embedding.model: {embed_model}")

    # ── Save ─────────────────────────────────────────────────────
    if changed:
        _save_config(cfg)
        log.info(f"💾 Đã cập nhật config → {_CONFIG_PATH}")
    else:
        log.info("✅ Tất cả cấu hình OK!")

    log.info("=" * 60)
    return cfg


if __name__ == "__main__":
    # Khi chạy trực tiếp, bật console logging để user thấy output
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    check_and_fix()
