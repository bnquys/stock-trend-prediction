"""Module Embedding: tạo và cache embedding vectors."""

import json
import random
import hashlib
import logging
from pathlib import Path
from stock_analysis.pplx_embed import PerplexityEmbeddingService

import numpy as np

embed_service = PerplexityEmbeddingService()

def get_embedding(text: str) -> np.ndarray:
    """Gọi API tạo embedding (thay thế bằng API thực tế). Trả về numpy array."""
    hash_string = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rng = random.Random(hash_string)
    return np.array([rng.random() for _ in range(2560)])


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
    vector = embed_service.get_embedding(response_text)
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
