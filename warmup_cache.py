"""
warmup_cache.py — Pre-generate Analysis Embeddings (CLI)
═══════════════════════════════════════════════════════════════════════
Chạy pipeline tuần tự cho tất cả windows → cache vào artifacts/embeddings/

Usage:
    uv run warmup_cache.py                    # Tất cả stocks
    uv run warmup_cache.py --target VNM       # Chỉ VNM
    uv run warmup_cache.py --target VNM FPT   # VNM + FPT
    uv run warmup_cache.py --skip-every 5     # Mỗi 5 ngày (nhanh hơn)

Có thể Ctrl+C bất cứ lúc nào — đã cache sẽ không gọi lại.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env trước khi import pipeline (module-level config)
load_dotenv(override=True)

from src.config import Config
from src.technical.preprocessor import load_csv
from src.fundamental.pipeline import pipeline


def warmup_stock(
    stock_id: str,
    csv_path: str,
    model: str,
    window: int = 20,
    skip_every: int = 1,
) -> tuple[int, int]:
    """
    Cache tất cả windows cho 1 stock.

    Returns:
        (success_count, error_count)
    """
    df = load_csv(csv_path)

    success_count = 0
    errors = 0
    error_msgs: list[str] = []

    steps = list(range(window, len(df), skip_every))
    pbar = tqdm(steps, desc=stock_id, unit="win")

    for step in pbar:
        start_idx = max(0, step - window + 1)
        date_end = pd.Timestamp(df["date"].iloc[step]).to_pydatetime()
        date_start = pd.Timestamp(df["date"].iloc[start_idx]).to_pydatetime()

        try:
            pipeline(
                model=model,
                stock_id=stock_id,
                date_start=date_start,
                date_end=date_end,
            )
            success_count += 1
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted! {stock_id}: {success_count} cached so far.")
            raise
        except Exception as e:
            logging.error(e)
            errors += 1
            if len(error_msgs) < 5:
                error_msgs.append(f"date={date_end.date()}: {e}")

        pbar.set_postfix(ok=success_count, err=errors)

    pbar.close()

    # Summary
    print(f"  ✓ {stock_id}: {success_count} ok, {errors} errors (total: {len(steps)})")
    if error_msgs:
        print(f"  Errors (first {len(error_msgs)}):")
        for msg in error_msgs:
            print(f"    • {msg}")

    return success_count, errors


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate analysis embeddings for training cache."
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=None,
        help="Stock IDs to cache (e.g., VNM FPT). Default: all from config.",
    )
    parser.add_argument(
        "--skip-every",
        type=int,
        default=1,
        help="Cache every N windows (1=all, 5=every 5 days). Default: 1",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    cfg = Config.load("configs/")
    window = cfg.env["window"]
    model = cfg.analysis["model"]
    paths = cfg.data["paths"]

    # Hiển thị model thực tế sẽ dùng (CKey có model riêng)
    from src.fundamental.llm_client import LLMClient
    _display_client = LLMClient(model=model)
    display_model = f"{_display_client.model} (backend: {_display_client.backend})"

    # Filter targets
    if args.target:
        targets = {t.upper() for t in args.target}
        paths = [p for p in paths if Path(p).stem.upper() in targets]
        if not paths:
            print(f"❌ No matching stocks for: {args.target}")
            print(f"   Available: {[Path(p).stem for p in cfg.data['paths']]}")
            sys.exit(1)

    print(f"═══ Warmup Cache ═══")
    print(f"  Model:      {display_model}")
    print(f"  Window:     {window}")
    print(f"  Skip every: {args.skip_every}")
    print(f"  Stocks:     {[Path(p).stem for p in paths]}")
    print()

    total_ok = 0
    total_err = 0

    for p in paths:
        stock_id = Path(p).stem
        if not Path(p).exists():
            print(f"  ✗ Skipping {stock_id}: {p} not found")
            continue

        df = load_csv(p)
        print(f"  {stock_id}: {len(df)} rows | {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

        try:
            ok, err = warmup_stock(stock_id, p, model, window, args.skip_every)
            total_ok += ok
            total_err += err
        except KeyboardInterrupt:
            print("\n⚠️ Stopped by user.")
            break

    print()
    print(f"═══ Done: {total_ok} cached, {total_err} errors ═══")


if __name__ == "__main__":
    main()
