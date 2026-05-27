"""
src/fundamental/llm_client.py
════════════════════════════════════════════════════════════════════════════
LLMClient — Gọi LLM API và cache response.

Hỗ trợ 2 backend:
  - "ckey"   : REST API (https://ckey.vn/v1/chat/completions)
  - "gradio" : Gradio client (self-hosted model)

Backend được chọn qua config.yaml → llm_backend
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import yaml
import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
load_dotenv(_PACKAGE_DIR / ".env")
load_dotenv(Path(".env"))  # Also check project root .env


# ─────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────

def _load_analysis_cfg() -> dict:
    cfg_path = _PACKAGE_DIR / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


_analysis_cfg = _load_analysis_cfg()


# ─────────────────────────────────────────────────────────────────────────
# Gradio client (lazy-init)
# ─────────────────────────────────────────────────────────────────────────

_gradio_client = None


def _get_gradio_client():
    """Lazy initialization of Gradio client — chỉ kết nối khi cache miss."""
    global _gradio_client
    if _gradio_client is None:
        from gradio_client import Client
        url = _analysis_cfg["llm"].get("gradio_url", "")
        if not url:
            url = input(
                "⚠️ llm.gradio_url trống. Nhập Gradio URL (chỉ dùng cho phiên này): "
            ).strip()
            if not url:
                raise ValueError("Gradio URL không được để trống!")
            _analysis_cfg["llm"]["gradio_url"] = url
        _gradio_client = Client(url)
    return _gradio_client


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class LLMClient:
    """
    LLM Client hỗ trợ 2 backend: CKey REST API và Gradio.

    Backend được tự động chọn từ config.yaml (llm_backend).
    Có thể override bằng tham số `backend` khi khởi tạo.
    """

    def __init__(
        self,
        model: str | None = None,
        backend: str | None = None,
    ):
        cfg = _analysis_cfg
        self.backend = backend or cfg.get("llm_backend", "ckey")

        if self.backend == "ckey":
            ckey_cfg = cfg.get("ckey", {})
            # CKey luôn dùng model từ config (model param chỉ dùng cho Gradio/hash)
            self.model = ckey_cfg.get("model", "mistral-small-4-119b-2603")
            self.url = ckey_cfg.get("url", "https://ckey.vn/v1/chat/completions")
            self.api_key = os.environ.get("CKEY_API", "")
            if not self.api_key:
                from getpass import getpass
                self.api_key = getpass(
                    "⚠️ CKEY_API chưa có. Nhập API key (chỉ dùng cho phiên này, "
                    "muốn persist hãy điền vào .env): "
                )
                if not self.api_key:
                    raise ValueError("CKEY_API không được để trống!")
                os.environ["CKEY_API"] = self.api_key
        elif self.backend == "gradio":
            llm_cfg = cfg.get("llm", {})
            self.model = model or llm_cfg.get("model", "")
            self.url = llm_cfg.get("gradio_url", "")
            if not self.url:
                self.url = input(
                    "⚠️ llm.gradio_url trống. Nhập Gradio URL (chỉ dùng cho phiên này, "
                    "muốn persist hãy điền vào src/fundamental/config.yaml): "
                ).strip()
                if not self.url:
                    raise ValueError("Gradio URL không được để trống!")
        else:
            raise ValueError(f"Backend không hợp lệ: '{self.backend}'. Chọn 'ckey' hoặc 'gradio'.")

        log.debug(f"[LLMClient] backend={self.backend} model={self.model}")

    @staticmethod
    def get_hash_id(model: str, report_hash_id: str) -> str:
        """Tạo hash_id từ model + report_hash_id."""
        content = f"{model}::{report_hash_id}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ─────────────────────────────────────────────────────────────────
    # Backend dispatch
    # ─────────────────────────────────────────────────────────────────

    def _call_api(self, prompt: str) -> str:
        """Gọi LLM API với retry. Dispatch theo self.backend."""
        if self.backend == "ckey":
            return self._call_ckey(prompt)
        else:
            return self._call_gradio(prompt)

    def _call_ckey(self, prompt: str) -> str:
        """Gọi CKey REST API (OpenAI-compatible)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.url, headers=headers, json=payload, timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                # OpenAI-compatible response format
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                log.warning(f"[CKey] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        raise RuntimeError(
            f"CKey API failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def _call_gradio(self, prompt: str) -> str:
        """Gọi Gradio API."""
        client = _get_gradio_client()
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = client.predict(
                    single_prompt_input=prompt,
                    api_name="/process_api",
                )
                return result
            except Exception as e:
                last_error = e
                log.warning(f"[Gradio] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        raise RuntimeError(
            f"Gradio LLM API failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    # ─────────────────────────────────────────────────────────────────
    # Main method: respond with caching
    # ─────────────────────────────────────────────────────────────────

    def respond(
        self,
        responses_dir: Path,
        report_hash_id: str,
        prompt: str,
        overwrite: bool = False,
    ) -> tuple[Path, str]:
        """
        Gọi LLM và lưu response vào responses/. Trả về (file_path, hash_id).
        Cache dựa trên hash(model, report_hash_id).
        """
        responses_dir.mkdir(parents=True, exist_ok=True)
        log_file = responses_dir / "logs.json"
        logs: dict = {}
        if log_file.exists():
            try:
                logs = json.loads(log_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logs = {}

        hash_id = self.get_hash_id(self.model, report_hash_id)

        # Check cache (chỉ dùng cache nếu respond_status là success)
        if not overwrite and hash_id in logs and logs[hash_id].get("respond_status") == "success":
            cached_file = responses_dir / f"{hash_id}.md"
            if cached_file.exists():
                log.debug(f"Response cache hit: {cached_file}")
                return cached_file, hash_id

        # Gọi API
        response_file = responses_dir / f"{hash_id}.md"
        try:
            text = self._call_api(prompt)
            response_file.write_text(text, encoding="utf-8")
            respond_status = "success"
        except Exception as e:
            log.error(f"LLM API failed: {e}")
            respond_status = "failed"

        # Cập nhật logs
        logs[hash_id] = {
            "model": self.model,
            "backend": self.backend,
            "report_hash_id": report_hash_id,
            "respond_path": str(response_file),
            "respond_status": respond_status,
            "created_date": datetime.now().isoformat(),
        }
        log_file.write_text(
            json.dumps(logs, indent=4, ensure_ascii=False), encoding="utf-8"
        )

        if respond_status == "failed":
            raise RuntimeError(
                f"LLM respond failed cho hash_id={hash_id}. Xem logs tại {log_file}"
            )

        return response_file, hash_id
