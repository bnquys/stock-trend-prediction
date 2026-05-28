"""
scripts/train.py — Entry point for training.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 500
    python scripts/train.py --config configs/
    python scripts/train.py --resume weights/ckpt_ep200.pkl
"""
from __future__ import annotations
import argparse, logging, os, sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import Config
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/logs/train.log", mode="w", encoding="utf-8"),
    ]
)
logging.getLogger("src.training").setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="RL Trading — Train")
    parser.add_argument("--config", default="configs/",
                        help="Path to config dir or single YAML file")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pkl to resume from")
    args = parser.parse_args()

    os.makedirs("artifacts/logs", exist_ok=True)
    cfg = Config.load(args.config)

    try:
        trainer = Trainer(cfg)
        trainer.run(n_episodes=args.episodes, resume_from=args.resume)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted. Last checkpoint saved.")
    except Exception as e:
        print(f"\n[!] Training crashed: {e}")
        raise


if __name__ == "__main__":
    main()
