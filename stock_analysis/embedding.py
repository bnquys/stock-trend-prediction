"""Module Embedding: tạo và cache embedding vectors."""

import json
import logging
import time
from pathlib import Path
# from stock_analysis.pplx_embed import PerplexityEmbeddingService

import numpy as np
from gradio_client import Client

# embed_service = PerplexityEmbeddingService()
client = Client("https://bd9302340272b38b66.gradio.live")

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def get_embedding(text: str) -> np.ndarray:
    """Gọi Gradio API để lấy embedding, có retry khi server lỗi."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.predict(
                text=text,
                api_name="/get_embedding",
            )
            return np.array(result)
        except Exception as e:
            last_error = e
            logging.warning(f"[Embedding] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"Embedding API failed after {MAX_RETRIES} attempts: {last_error}") from last_error

def embed_response(responses_dir: Path, response_hash_id: str, response_text: str, overwrite: bool = False) -> np.ndarray:
    """
    Tạo embedding cho response và lưu vào embeddings.npz (một file duy nhất).
    Cập nhật embed_key trong responses/logs.json.
    Trả về numpy array của vector.
    """
    npz_path = responses_dir / "embeddings.npz"

    # Load existing vectors nếu có
    existing = {}
    if npz_path.exists():
        with np.load(npz_path) as data:
            existing = {k: data[k] for k in data.files}

    # Check cache: nếu vector đã tồn tại và không overwrite
    if not overwrite and response_hash_id in existing:
        logging.info(f"Embedding đã tồn tại cho {response_hash_id}")
        return existing[response_hash_id]

    # Gọi API embedding
    # vector = embed_service.get_embedding(response_text)
    vector = get_embedding(response_text)
    existing[response_hash_id] = vector
    np.savez(npz_path, **existing)

    # Cập nhật embed_key trong logs
    log_file = responses_dir / "logs.json"
    logs = {}
    if log_file.exists():
        try:
            logs = json.loads(log_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logs = {}

    if response_hash_id in logs:
        logs[response_hash_id]["embed_key"] = response_hash_id
        log_file.write_text(json.dumps(logs, indent=4, ensure_ascii=False), encoding="utf-8")

    return vector
