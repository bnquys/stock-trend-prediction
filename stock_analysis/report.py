"""Module Report: tạo báo cáo tổng hợp từ dữ liệu CSV."""

import json
import hashlib
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from stock_analysis.data import Data


class Report:
    """Tạo báo cáo tổng hợp từ dữ liệu đã tải của một mã chứng khoán."""

    def __init__(self, data: Data):
        self.data = data
        self.data_dir = data.get_directory()

    def _read_csv(self, filename: str) -> pd.DataFrame:
        """Đọc file CSV từ thư mục dữ liệu."""
        return pd.read_csv(self.data_dir / filename)

    def _df_to_section(self, df: pd.DataFrame) -> str:
        """Chuyển DataFrame thành markdown, trả về thông báo nếu rỗng."""
        if not df.empty:
            return df.to_markdown(index=False)
        return "**Không có dữ liệu.**\n\n --- \n\n"

    def _filter_by_date(self, df: pd.DataFrame, date_col: str, date_start: datetime, date_end: datetime) -> pd.DataFrame:
        """Lọc DataFrame theo khoảng thời gian dựa trên cột ngày."""
        dates = pd.to_datetime(df[date_col], errors="coerce")
        return df[dates.between(date_start, date_end)]

    def _filter_by_quarter(self, df: pd.DataFrame, period_col: str, q_start: pd.Period, q_end: pd.Period) -> pd.DataFrame:
        """Lọc DataFrame theo khoảng quý dựa trên cột period."""
        periods = pd.PeriodIndex(df[period_col], freq="Q")
        mask = (periods >= q_start) & (periods <= q_end)
        return df[mask]

    @staticmethod
    def get_hash_id(stock_id: str, date_start: datetime, date_end: datetime) -> str:
        """Tạo hash_id từ stock_id + date_start + date_end."""
        content = f"{stock_id}::{date_start.isoformat()}::{date_end.isoformat()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def create(self, date_start: datetime, date_end: datetime) -> tuple[Path, str]:
        """Tạo báo cáo tổng hợp. Trả về (đường dẫn file .md, hash_id)."""
        if date_start > date_end:
            raise ValueError("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.")
        if date_end > datetime.now():
            raise ValueError("Ngày kết thúc không được lớn hơn ngày hiện tại.")

        stock_id = self.data.id
        hash_id = self.get_hash_id(stock_id, date_start, date_end)

        # Kiểm tra cache trong reports/
        reports_dir = self.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        log_file = reports_dir / "logs.json"
        logs = {}
        if log_file.exists():
            try:
                logs = json.loads(log_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logs = {}

        # Nếu đã có report với hash_id này, trả về file cũ
        if hash_id in logs:
            cached_path = Path(logs[hash_id]["file_path"])
            if cached_path.exists():
                logging.info(f"Report đã tồn tại: {cached_path}")
                return cached_path, hash_id

        q_start = pd.Period(date_start, freq="Q")
        q_end = pd.Period(date_end, freq="Q")

        sections = []
        sections.append(f"# Tổng hợp thông tin của {stock_id} từ ngày {date_start} đến {date_end}.\n\n")

        # 1. Thông tin công ty
        sections.append("\n\n## 1. Thông tin công ty (Company Info)\n\n")
        df_tmp = self._read_csv("company_info.csv")
        sections.append(self._df_to_section(df_tmp))

        # 2. Cổ đông lớn
        sections.append("\n\n## 2. Danh sách cổ đông lớn (Major Shareholders)\n\n")
        shareholders_df = self._read_csv("shareholders.csv")
        df_tmp = self._filter_by_date(shareholders_df, "date", date_start, date_end)
        sections.append(self._df_to_section(df_tmp))

        # 3. Ban lãnh đạo
        sections.append("\n\n## 3. Ban lãnh đạo / Quản lý cấp cao (Officers)\n\n")
        df_tmp = self._read_csv("officers.csv")
        sections.append(self._df_to_section(df_tmp))

        # 4. Công ty con
        sections.append("\n\n## 4. Các công ty con (Subsidiaries)\n\n")
        df_tmp = self._read_csv("subsidiaries.csv")
        sections.append(self._df_to_section(df_tmp))

        # 5. Tin tức
        sections.append("\n\n## 5. Các tin tức liên quan (News)\n\n")
        news_df = self._read_csv("news.csv")
        df_tmp = self._filter_by_date(news_df, "public_date", date_start, date_end)
        sections.append(self._df_to_section(df_tmp))

        # 6. Sự kiện
        sections.append("\n\n## 6. Các sự kiện liên quan (Events)\n\n")
        events_df = self._read_csv("events.csv")
        df_tmp = self._filter_by_date(events_df, "public_date", date_start, date_end)
        sections.append(self._df_to_section(df_tmp))

        # 7. Báo cáo kết quả kinh doanh
        sections.append("\n\n## 7. Báo cáo kết quả kinh doanh (Income Statement)\n\n")
        income_df = self._read_csv("income_statement.csv")
        df_tmp = self._filter_by_quarter(income_df, "period", q_start, q_end)
        sections.append(self._df_to_section(df_tmp))

        # 8. Bảng cân đối kế toán
        sections.append("\n\n## 8. Bảng cân đối kế toán (Balance Sheet)\n\n")
        balance_df = self._read_csv("balance_sheet.csv")
        df_tmp = self._filter_by_quarter(balance_df, "period", q_start, q_end)
        sections.append(self._df_to_section(df_tmp))

        # 9. Lưu chuyển tiền tệ
        sections.append("\n\n## 9. Báo cáo lưu chuyển tiền tệ (Cash Flow)\n\n")
        cash_df = self._read_csv("cash_flow.csv")
        df_tmp = self._filter_by_quarter(cash_df, "period", q_start, q_end)
        sections.append(self._df_to_section(df_tmp))

        # 10. Chỉ số tài chính
        sections.append("\n\n## 10. Các chỉ số tài chính (Financial Ratios)\n\n")
        ratio_df = self._read_csv("financial_ratios.csv")
        df_tmp = self._filter_by_quarter(ratio_df, "period", q_start, q_end)
        sections.append(self._df_to_section(df_tmp))

        # 11. Thuyết minh BCTC
        sections.append("\n\n## 11. Trích thuyết minh báo cáo tài chính (Financial Disclosures/Notes)\n\n")
        notes_df = self._read_csv("financial_notes.csv").copy()
        period_clean = notes_df["report_period"].astype(str) \
            .str.replace(r"^\d{4}$", lambda m: m.group(0) + "-Q4", regex=True)
        periods = pd.PeriodIndex(period_clean, freq="Q")
        mask = (periods >= q_start) & (periods <= q_end)
        df_tmp = notes_df[mask]
        sections.append(self._df_to_section(df_tmp))

        # 12. Sức khỏe tài chính
        sections.append("\n\n## 12. Điểm đánh giá sức khỏe tài chính (Financial Health Score)\n\n")
        health_df = self._read_csv("financial_health.csv")
        df_tmp = self._filter_by_quarter(health_df, "period", q_start, q_end)
        sections.append(self._df_to_section(df_tmp))

        # Ghi file báo cáo vào reports/
        context = "".join(sections)
        file_report = reports_dir / f"{hash_id}.md"
        file_report.write_text(context, encoding="utf-8")

        # Cập nhật logs
        logs[hash_id] = {
            "date_start": date_start.isoformat(),
            "date_end": date_end.isoformat(),
            "file_path": str(file_report),
            "created_date": datetime.now().isoformat(),
        }
        log_file.write_text(json.dumps(logs, indent=4, ensure_ascii=False), encoding="utf-8")

        return file_report, hash_id
