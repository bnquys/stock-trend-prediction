"""
src/fundamental/cache.py
════════════════════════════════════════════════════════════════════════════
EmbeddingCache — Pre-load tất cả embeddings vào RAM để training zero I/O.

Thay vì mỗi step gọi pipeline() → load npz → tìm key → trả vector,
class này load toàn bộ 1 lần khi khởi tạo.

Usage:
    cache = EmbeddingCache(["VNM", "FPT", "HPG", "VIC"])
    vector = cache.get(stock_id="VNM", date_start=..., date_end=...)
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent


class EmbeddingCache:
    """
    Pre-load tất cả embeddings cho các stocks vào RAM.
    Lookup bằng (stock_id, date_start, date_end) → numpy vector.
    """

    def __init__(self, stock_ids: list[str], base_dir: Path | None = None):
        """
        Args:
            stock_ids: Danh sách mã cổ phiếu cần load (VNM, FPT, ...)
            base_dir: Thư mục gốc chứa embeddings/ (mặc định: artifacts/embeddings/)
        """
        self._base_dir = base_dir or (_PROJECT_ROOT / "artifacts" / "embeddings")
        self._vectors: dict[str, dict[str, np.ndarray]] = {}  # {stock_id: {hash_id: vector}}
        self._report_map: dict[str, dict[str, str]] = {}  # {stock_id: {report_hash: response_hash}}
        self._report_logs: dict[str, dict] = {}  # {stock_id: report logs.json content}
        self._response_logs: dict[str, dict] = {}  # {stock_id: response logs.json content}

        total_vectors = 0
        for sid in stock_ids:
            n = self._load_stock(sid)
            total_vectors += n

        log.debug(f"[EmbeddingCache] Loaded {total_vectors} vectors for {len(stock_ids)} stocks into RAM")

    def _load_stock(self, stock_id: str) -> int:
        """Load tất cả embeddings + logs cho 1 stock. Trả về số vectors loaded."""
        stock_dir = self._base_dir / stock_id
        responses_dir = stock_dir / "responses"
        reports_dir = stock_dir / "reports"

        # Load embeddings.npz
        npz_path = responses_dir / "embeddings.npz"
        if not npz_path.exists():
            log.warning(f"[EmbeddingCache] {stock_id}: embeddings.npz not found — no cached vectors")
            self._vectors[stock_id] = {}
            return 0

        with np.load(npz_path, allow_pickle=True) as data:
            vectors = {k: data[k].astype(np.float32) for k in data.files}
        self._vectors[stock_id] = vectors

        # Load response logs (response_hash → report_hash mapping)
        resp_log_path = responses_dir / "logs.json"
        if resp_log_path.exists():
            try:
                self._response_logs[stock_id] = json.loads(
                    resp_log_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                self._response_logs[stock_id] = {}
        else:
            self._response_logs[stock_id] = {}

        # Build reverse map: report_hash → response_hash (for lookup)
        report_to_response: dict[str, str] = {}
        for resp_hash, info in self._response_logs[stock_id].items():
            rpt_hash = info.get("report_hash_id", "")
            if rpt_hash:
                report_to_response[rpt_hash] = resp_hash
        self._report_map[stock_id] = report_to_response

        # Load report logs (date_hash → report_hash mapping)
        rpt_log_path = reports_dir / "logs.json"
        if rpt_log_path.exists():
            try:
                self._report_logs[stock_id] = json.loads(
                    rpt_log_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                self._report_logs[stock_id] = {}
        else:
            self._report_logs[stock_id] = {}

        return len(vectors)

    def get(
        self,
        stock_id: str,
        date_start: datetime,
        date_end: datetime,
    ) -> np.ndarray | None:
        """
        Lookup embedding vector cho (stock_id, date_start, date_end).
        Trả về numpy array (embed_dim,) hoặc None nếu không tìm thấy.
        """
        if stock_id not in self._vectors:
            return None

        # Tìm report hash từ date range (giống logic trong Report.create)
        date_hash = self._compute_date_hash(stock_id, date_start, date_end)
        if date_hash is None:
            return None

        # date_hash → report_hash (từ report logs)
        report_logs = self._report_logs.get(stock_id, {})
        report_info = report_logs.get(date_hash)
        if report_info is None:
            return None

        report_hash = report_info.get("content_hash", date_hash)

        # report_hash → response_hash
        report_map = self._report_map.get(stock_id, {})
        response_hash = report_map.get(report_hash)
        if response_hash is None:
            return None

        # response_hash → vector
        vectors = self._vectors.get(stock_id, {})
        vector = vectors.get(response_hash)
        return vector

    def _compute_date_hash(
        self, stock_id: str, date_start: datetime, date_end: datetime
    ) -> str | None:
        """
        Tính date_hash giống Report.get_hash_id().
        Format: sha256(f"{stock_id}::{date_start.isoformat()}::{date_end.isoformat()}")
        """
        import hashlib

        content = f"{stock_id}::{date_start.isoformat()}::{date_end.isoformat()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @property
    def stats(self) -> dict[str, int]:
        """Trả về thống kê số vectors per stock."""
        return {sid: len(vecs) for sid, vecs in self._vectors.items()}
