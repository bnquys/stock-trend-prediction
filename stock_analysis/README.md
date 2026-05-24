# Stock Analysis Pipeline

Pipeline phân tích chứng khoán: tải dữ liệu → tạo report → gọi LLM phân tích → tạo embedding vector.

## Cấu trúc package

```
stock_analysis/
├── __init__.py
├── data.py                # Class Data: quản lý dữ liệu CSV
├── report.py              # Class Report: tạo báo cáo markdown
├── llm_client.py          # Class LLMClient: gọi LLM API
├── embedding.py           # Hàm tạo embedding vector
├── pipeline.py            # Hàm pipeline: orchestrate toàn bộ
├── prompt.txt             # Prompt template cho LLM
├── .env                   # API key (không commit lên git)
├── README.md              # Tài liệu hướng dẫn
└── st_data/               # Thư mục dữ liệu (tự tạo khi chạy downloader)
    └── VNM/
        ├── logs.json          # Log trạng thái download
        ├── *.csv              # Dữ liệu thô
        ├── reports/           # Cache báo cáo markdown
        │   ├── logs.json
        │   └── {hash_id}.md
        └── responses/         # Cache LLM responses + embeddings
            ├── logs.json
            ├── {hash_id}.md
            └── embeddings.npz
```

Tất cả đường dẫn trong package đều sử dụng `Path(__file__).parent` nên có thể di chuyển folder `stock_analysis/` đến bất kỳ đâu mà vẫn hoạt động bình thường.

## Cài đặt

```bash
pip install -r pyproject.toml
```

Tạo file `.env` bên trong `stock_analysis/` với API key:

```
API_KEY=your_llm_api_key_here
```

## Sử dụng

### Bước 1: Tải dữ liệu (chỉ cần chạy 1 lần hoặc khi cần cập nhật)

Yêu cầu: môi trường có `vnstock_data` đã login.

```bash
uv run downloader.py VNM
uv run downloader.py VNM --overwrite   # ghi đè dữ liệu cũ
```

### Bước 2: Chạy pipeline để lấy embedding vector

```python
from datetime import datetime
from stock_analysis import pipeline

result = pipeline(
    model="mistral-small-4-119b-2603",
    stock_id="VNM",
    date_start=datetime(2024, 5, 1),
    date_end=datetime(2026, 5, 20),
)

# Kết quả
vector = result["vector"]              # numpy array (2560,)
report_file = result["report_file"]    # Path to report .md
response_file = result["response_file"]  # Path to LLM response .md
```

### Debug: xem nội dung report và LLM response

```python
from IPython.display import Markdown, display

# Xem report tổng hợp dữ liệu
display(Markdown(result["report_file"].read_text(encoding="utf-8")))

# Xem phản hồi phân tích từ LLM
display(Markdown(result["response_file"].read_text(encoding="utf-8")))

# Xem vector embedding
print(result["vector"].shape)  # (2560,)
```

### Chạy lại với overwrite (bỏ cache)

```python
result = pipeline(
    model="mistral-small-4-119b-2603",
    stock_id="VNM",
    date_start=datetime(2024, 5, 1),
    date_end=datetime(2026, 5, 20),
    overwrite=True,  # xoá cache, gọi lại LLM + embedding
)
```

## Cơ chế cache (tránh gọi API thừa)

| Bước | Cache key | Gọi API khi nào? |
|------|-----------|-------------------|
| Report | `hash(stock_id, date_start, date_end)` | Chưa có file `.md` trong `reports/` |
| LLM | `hash(model, report_hash_id)` | Chưa có response hoặc `respond_status != "success"` |
| Embedding | `response_hash_id` trong `embeddings.npz` | Key chưa có trong `.npz` |

Chạy lại pipeline cùng tham số → **0 API calls** (tất cả hit cache local).

## Sử dụng từng module riêng lẻ

```python
from stock_analysis import Data, Report, LLMClient, embed_response

# Chỉ tạo report
data = Data("VNM")
report_file, report_hash_id = Report(data).create(
    datetime(2024, 5, 1), datetime(2026, 5, 20)
)

# Chỉ gọi LLM
llm = LLMClient(model="mistral-small-4-119b-2603")
response_file, response_hash_id = llm.respond(
    responses_dir=data.get_directory() / "responses",
    report_hash_id=report_hash_id,
    prompt="Phân tích...",
)
```
