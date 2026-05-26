"""
train.py — Thin entry point for headless training.
════════════════════════════════════════════════════════════════════════════
Delegates all logic to src/training/trainer.py (Trainer class).

Chạy:
    python train.py
    python train.py --episodes 500
    python train.py --resume artifacts/weights/ckpt_ep100.pkl
    python train.py --mode charts
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.logging_config import setup_logging
setup_logging()
log = logging.getLogger(__name__)

from src.config import Config
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="RL Trading — Train & Evaluate")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="train", choices=["train", "charts"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pkl to resume training")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    trainer = Trainer(cfg)

    if args.mode == "train":
        try:
            history = trainer.run(n_episodes=args.episodes, resume_from=args.resume)
            trainer.evaluate(history=history)
        except KeyboardInterrupt:
            log.info("\n[!] Training interrupted. Last checkpoint saved.")
        except Exception as e:
            log.error(f"\n[!] Training crashed: {e}", exc_info=True)
            raise
    else:
        # Charts-only mode: evaluate from existing best_model
        trainer.evaluate()


if __name__ == "__main__":
    main()
