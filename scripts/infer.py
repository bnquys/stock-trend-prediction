"""
scripts/infer.py — Entry point for inference.

Usage:
    python scripts/infer.py --stock data/VNM.csv
    python scripts/infer.py --all
    python scripts/infer.py --model weights/best_model.pkl --stock data/FPT.csv
"""
from __future__ import annotations
import argparse, logging, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import Config
from src.inference.inferencer import Inferencer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="RL Trading — Inference")
    parser.add_argument("--config", default="configs/",
                        help="Path to config dir or single YAML file")
    parser.add_argument("--model", default="weights/best_model.pkl",
                        help="Path to trained model")
    parser.add_argument("--scaler", default="weights/scaler.pkl",
                        help="Path to fitted scaler")
    parser.add_argument("--stock", type=str, default=None,
                        help="Path to single stock CSV")
    parser.add_argument("--all", action="store_true",
                        help="Run inference on all stocks in config")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    infer = Inferencer(cfg, model_path=args.model, scaler_path=args.scaler)

    if args.all:
        paths = cfg.data.get("paths", [])
        stock_map = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
        results = infer.predict_batch(stock_map)
        print(f"\n{'═' * 70}")
        print(f"{'Stock':<8} {'Action':<12} {'Confidence':>10} {'Close':>12} {'Date':<12}")
        print(f"{'─' * 70}")
        for symbol, r in results.items():
            if "error" in r:
                print(f"{symbol:<8} ERROR: {r['error']}")
            else:
                print(f"{symbol:<8} {r['action_name']:<12} {r['confidence']:>10.2%} "
                      f"{r['latest_close']:>12,.0f} {r['latest_date']:<12}")
        print(f"{'═' * 70}")

    elif args.stock:
        result = infer.predict(args.stock)
        print(f"\n{'═' * 50}")
        print(f"  Stock:      {args.stock}")
        print(f"  Date:       {result['latest_date']}")
        print(f"  Close:      {result['latest_close']:,.0f} VND")
        print(f"  Action:     {result['action_name']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Q-values:   {result['q_values']}")
        print(f"{'═' * 50}")

    else:
        parser.print_help()
        print("\nSpecify --stock <path> or --all")


if __name__ == "__main__":
    main()
