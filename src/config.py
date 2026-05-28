"""
src/config.py
════════════════════════════════════════════════════════════════════════════
ConfigLoader — Merge multiple YAML config files into a single Config object.

Supports:
  - Modular configs (base.yaml, env.yaml, agent.yaml, training.yaml)
  - Single legacy config (config.yaml) for backward compatibility
  - CLI overrides via dot-notation (e.g., --set agent.lr=0.001)
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, return empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """Unified config object with typed sections."""

    project: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    @classmethod
    def from_dir(cls, config_dir: str | Path = "configs/") -> "Config":
        """
        Load modular configs from a directory.
        Files: base.yaml, env.yaml, agent.yaml, training.yaml
        Analysis config (dim, projection) lives in src/fundamental/config.yaml.
        """
        d = Path(config_dir)
        base = _load_yaml(d / "base.yaml")

        return cls(
            project=base.get("project", {}),
            data=base.get("data", {}),
            split=base.get("split", {}),
            env=_load_yaml(d / "env.yaml"),
            agent=_load_yaml(d / "agent.yaml"),
            analysis=base.get("analysis", {}),
            training=_load_yaml(d / "training.yaml"),
            output=base.get("output", {}),
        )

    @classmethod
    def from_single(cls, path: str | Path = "configs/config.yaml") -> "Config":
        """
        Backward compat: load from a single monolithic config.yaml.
        Maps old structure to new Config sections.
        """
        cfg = _load_yaml(Path(path))
        return cls(
            project=cfg.get("project", {}),
            data=cfg.get("data", {}),
            split=cfg.get("split", {}),
            env=cfg.get("env", {}),
            agent=cfg.get("agent", {}),
            analysis=cfg.get("analysis", {}),
            training=cfg.get("training", {}),
            output=cfg.get("output", {}),
        )

    @classmethod
    def load(cls, path_or_dir: str | Path = "configs/") -> "Config":
        """
        Auto-detect: if path is a directory → from_dir(), if file → from_single().
        """
        p = Path(path_or_dir)
        if p.is_dir():
            return cls.from_dir(p)
        elif p.is_file():
            return cls.from_single(p)
        else:
            raise FileNotFoundError(f"Config path not found: {p}")

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dict (legacy format) for backward compat."""
        return {
            "project": self.project,
            "data": self.data,
            "split": self.split,
            "env": self.env,
            "agent": self.agent,
            "analysis": self.analysis,
            "training": self.training,
            "output": self.output,
        }

    def override(self, key: str, value: Any) -> None:
        """
        Override a config value using dot-notation.
        Example: config.override("agent.lr", 0.001)
        """
        parts = key.split(".")
        if len(parts) == 2:
            section, param = parts
            getattr(self, section)[param] = value
        elif len(parts) == 1:
            # Top-level key — try to find which section it belongs to
            for section_name in ["env", "agent", "analysis", "training", "output"]:
                section = getattr(self, section_name)
                if key in section:
                    section[key] = value
                    return
