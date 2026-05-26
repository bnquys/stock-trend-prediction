"""Module LLMClient: gọi LLM API và cache response."""

import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from random import randint

log = logging.getLogger(__name__)

import yaml
import requests
from dotenv import load_dotenv

_PACKAGE_DIR = Path(__file__).parent
load_dotenv(_PACKAGE_DIR / ".env")

# Load URL từ stock_analysis/config.yaml
def _load_analysis_cfg() -> dict:
    cfg_path = _PACKAGE_DIR / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

_analysis_cfg = _load_analysis_cfg()

# Lazy-init: chỉ tạo Gradio client khi thực sự cần gọi API (cache miss)
_client = None

def _get_client():
    """Lazy initialization of Gradio client — chỉ kết nối khi cache miss."""
    global _client
    if _client is None:
        from gradio_client import Client
        _client = Client(_analysis_cfg["llm"]["gradio_url"])
    return _client

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

class LLMClient:
    def __init__(
        self,
        model: str,
        url: str = "https://ckey.vn/v1/chat/completions",
    ):
        self.url = url
        self.model = model

    @staticmethod
    def get_hash_id(model: str, report_hash_id: str) -> str:
        """Tạo hash_id từ model + report_hash_id."""
        content = f"{model}::{report_hash_id}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _call_api(self, prompt: str) -> str:
        """Gọi LLM API với retry khi server lỗi, trả về nội dung text response."""
        client = _get_client()
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
                log.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
        raise RuntimeError(f"LLM API failed after {MAX_RETRIES} attempts: {last_error}") from last_error
        

    def respond(self, responses_dir: Path, report_hash_id: str, prompt: str, overwrite: bool = False) -> tuple[Path, str]:
        """
        Gọi LLM và lưu response vào responses/. Trả về (file_path, hash_id).
        Cache dựa trên hash(model, report_hash_id).
        """
        responses_dir.mkdir(parents=True, exist_ok=True)
        log_file = responses_dir / "logs.json"
        logs = {}
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
            "report_hash_id": report_hash_id,
            "respond_path": str(response_file),
            "respond_status": respond_status,
            "created_date": datetime.now().isoformat(),
        }
        log_file.write_text(json.dumps(logs, indent=4, ensure_ascii=False), encoding="utf-8")

        if respond_status == "failed":
            raise RuntimeError(f"LLM respond failed cho hash_id={hash_id}. Xem logs tại {log_file}")

        return response_file, hash_id
