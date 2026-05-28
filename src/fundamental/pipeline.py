"""Module Pipeline: orchestrate Report → LLM → Embedding."""

import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

from src.fundamental.data import Data
from src.fundamental.report import Report
from src.fundamental.llm_client import LLMClient
from src.fundamental.embedding import embed_response

_PACKAGE_DIR = Path(__file__).parent


def pipeline(
    model: str,
    stock_id: str,
    date_start: datetime,
    date_end: datetime,
    overwrite: bool = False,
) -> dict:
    """
    Pipeline: Report → LLM Analysis → Embedding.

    Yêu cầu: CSV đã có sẵn trong st_data/{stock_id}/
    (Dùng `python downloader.py {stock_id}` để tải trước nếu cần)

    Returns:
        dict với keys: report_file, response_file, vector
    """
    data = Data(stock_id)

    # Nếu overwrite, xoá cache reports/responses cũ
    if overwrite:
        data.invalidate_cache()

    # 1. Tạo report từ dữ liệu CSV có sẵn
    report_file, report_hash_id = Report(data).create(date_start, date_end)
    log.debug(f"Report: {report_file}")

    # 2. Gọi LLM để phân tích report (chỉ gửi body, bỏ tiêu đề placeholder)
    report_content = report_file.read_text(encoding="utf-8")
    # Bỏ dòng tiêu đề template (dòng đầu tiên bắt đầu bằng "# ")
    body_lines = report_content.split("\n")
    body = "\n".join(line for line in body_lines if not line.startswith("# Tổng hợp thông tin của {stock_id}"))
    prompt = (_PACKAGE_DIR / "prompt.txt").read_text(encoding="utf-8")
    prompt += "\n\n" + body

    responses_dir = data.get_directory() / "responses"
    llm = LLMClient(model=model)
    response_file, response_hash_id = llm.respond(
        responses_dir=responses_dir,
        report_hash_id=report_hash_id,
        prompt=prompt,
        overwrite=overwrite,
    )
    log.debug(f"LLM Response: {response_file}")

    # 3. Tạo embedding từ nội dung response
    response_text = response_file.read_text(encoding="utf-8")
    vector = embed_response(responses_dir, response_hash_id, response_text, overwrite=overwrite)
    log.debug(f"Embedding vector length: {len(vector)}")

    return {
        "report_file": report_file,
        "response_file": response_file,
        "vector": vector,
    }
