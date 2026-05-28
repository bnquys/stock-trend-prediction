"""
scripts/train.py — Entry point for training.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 500
    python scripts/train.py --config configs/
    python scripts/train.py --resume outputs/output_xxx/weights/ckpt_ep200.pkl
    python scripts/train.py --charts-only outputs/output_xxx/
"""
from __future__ import annotations
import argparse, logging, os, sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.logging_config import setup_logging
from src.config import Config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="RL Trading — Train")
    parser.add_argument("--config", default="configs/",
                        help="Path to config dir or single YAML file")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pkl to resume from")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip test evaluation after training")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    setup_logging()
    cfg = Config.load(args.config)

    try:
        trainer = Trainer(cfg)
        trainer.run(n_episodes=args.episodes, resume_from=args.resume)

        # Evaluate on test set
        test_results = None
        if not args.no_eval:
            test_results = trainer.evaluate()
            trainer.export_roi_table(test_results)

        # Generate charts
        if not args.no_charts:
            trainer.generate_charts(test_results)

        print(f"\n✅ Output directory: {trainer.run_dir}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted. Last checkpoint saved.")
    except Exception as e:
        print(f"\n[!] Training crashed: {e}")
        raise


if __name__ == "__main__":
    main()
