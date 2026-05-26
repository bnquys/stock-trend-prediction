"""Module Data: đại diện cho dữ liệu CSV của một mã chứng khoán."""

import json
import shutil
import logging
from pathlib import Path

log = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).parent


class Data:
    """Đại diện cho dữ liệu của một mã chứng khoán. Không phụ thuộc vnstock_data."""

    def __init__(self, id: str, root: Path = Path("artifacts/embeddings")):
        self.id = id
        self.root = root
        self._validate_data()

    def get_directory(self) -> Path:
        return self.root / self.id

    def _validate_data(self):
        """Kiểm tra dữ liệu dựa trên logs.json của downloader."""
        dir = self.get_directory()
        log_file = dir / "logs.json"

        if not dir.exists() or not log_file.exists():
            raise FileNotFoundError(
                f"Chưa có dữ liệu cho '{self.id}'. "
                f"Chạy 'python downloader.py {self.id}' để tải dữ liệu trước."
            )

        try:
            logs = json.loads(log_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise FileNotFoundError(
                f"File logs.json trong '{dir}' bị hỏng. "
                f"Chạy 'python downloader.py {self.id} --overwrite' để tải lại."
            )

        failed = [k for k, v in logs.items() if v.get("status") != "success"]
        if failed:
            raise FileNotFoundError(
                f"Dữ liệu chưa đầy đủ cho '{self.id}'. "
                f"Các mục lỗi: {', '.join(failed)}. "
                f"Chạy 'python downloader.py {self.id}' để tải lại."
            )

    def invalidate_cache(self):
        """Xoá reports/ và responses/ khi dữ liệu CSV được cập nhật."""
        dir = self.get_directory()
        for sub in ["reports", "responses"]:
            sub_dir = dir / sub
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
                log.debug(f"Invalidated cache: {sub_dir}")
