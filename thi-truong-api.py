"""
thi-truong-api.py — Entry-point để chạy FastAPI server cho thi-truong.html.

Usage:
    uv run python thi-truong-api.py
    # → Server chạy tại http://localhost:8000
    # → Mở thi-truong.html (file://) — trang tự động kết nối localhost:8000/api
"""
from __future__ import annotations
import uvicorn


def main() -> None:
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()