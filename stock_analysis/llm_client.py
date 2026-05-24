"""Module LLMClient: gọi LLM API và cache response."""

import json
import os
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

_PACKAGE_DIR = Path(__file__).parent
load_dotenv(_PACKAGE_DIR / ".env")


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
        """Gọi LLM API, trả về nội dung text response."""
        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            response = requests.post(self.url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if attempt < max_retries:
                time.sleep(2)

        raise RuntimeError(
            f"API call failed after {max_retries} retries. "
            f"Last status: {response.status_code}, Body: {response.text}"
        )

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
                logging.info(f"Response đã tồn tại: {cached_file}")
                return cached_file, hash_id

        # Gọi API
        response_file = responses_dir / f"{hash_id}.md"
        try:
            text = self._call_api(prompt)
            response_file.write_text(text, encoding="utf-8")
            respond_status = "success"
        except Exception as e:
            logging.error(f"LLM API failed: {e}")
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
