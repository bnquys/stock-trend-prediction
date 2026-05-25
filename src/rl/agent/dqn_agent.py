"""
src/rl/agent/dqn_agent.py
════════════════════════════════════════════════════════════════════════════
Double Dueling DQN Agent — PyTorch Implementation
Hỗ trợ Analysis Embedding (phân tích cơ bản) qua Addition fusion.
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, pickle, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

log = logging.getLogger(__name__)

# Tự động chọn Device (GPU nếu có)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingQNet(nn.Module):
    """
    Dueling DQN Architecture với optional Analysis Embedding fusion:
    Input → Shared Layers → features
    (Optional) Analysis Embed → Projection (2560→1024→512→128) → proj
    features (+) proj → (Value Stream & Advantage Stream) → Q(s,a)
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: list[int],
                 analysis_embed_dim: int | None = None,
                 analysis_proj_layers: list[int] | None = None):
        super().__init__()

        # Shared Feature Extraction
        layers = []
        prev = obs_size
        for h in hidden[:-1]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            prev = h
        self.feature_layer = nn.Sequential(*layers)

        # ── Analysis Embedding Projection (optional) ──────────────────
        # Giảm chiều từ từ: 2560 → 1024 → 512 → 128 (= prev = hidden[-2])
        self.has_analysis = analysis_embed_dim is not None
        if self.has_analysis and analysis_embed_dim is not None:
            proj_dims = analysis_proj_layers or [1024, 512, prev]
            proj_layers = []
            in_dim: int = analysis_embed_dim
            for out_dim in proj_dims:
                proj_layers.append(nn.Linear(in_dim, out_dim))
                proj_layers.append(nn.ReLU())
                proj_layers.append(nn.LayerNorm(out_dim))
                in_dim = out_dim
            self.analysis_proj = nn.Sequential(*proj_layers)
            log.debug(f"[Net] Analysis projection: {analysis_embed_dim} → {proj_dims}")

        # Value Stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(prev, hidden[-1]),
            nn.ReLU(),
            nn.Linear(hidden[-1], 1)
        )

        # Advantage Stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev, hidden[-1]),
            nn.ReLU(),
            nn.Linear(hidden[-1], n_actions)
        )

    def forward(self, x: torch.Tensor, analysis_embed: torch.Tensor | None = None) -> torch.Tensor:
        features = self.feature_layer(x)

        # Addition fusion: cộng projected analysis embedding vào features
        if self.has_analysis and analysis_embed is not None:
            proj = self.analysis_proj(analysis_embed)
            features = features + proj

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q


class ReplayBuffer:
    """Replay buffer hỗ trợ lưu thêm analysis_embed (optional)."""
    def __init__(self, cap: int = 20_000, has_analysis: bool = False):
        self._buf: deque = deque(maxlen=cap)
        self.has_analysis = has_analysis

    def push(self, obs, action, reward, next_obs, done, analysis_embed=None):
        entry = (
            obs.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            float(done),
        )
        if self.has_analysis:
            embed = analysis_embed.astype(np.float32) if analysis_embed is not None else None
            entry = entry + (embed,)
        self._buf.append(entry)

    def sample(self, bs: int):
        idx = np.random.choice(len(self._buf), bs, replace=False)
        batch = [self._buf[i] for i in idx]

        if self.has_analysis:
            obs_b, act_b, rew_b, nobs_b, done_b, embed_b = zip(*batch)
            # analysis_embed batch
            embed_tensor = None
            if embed_b[0] is not None:
                embed_tensor = torch.tensor(np.array(embed_b), dtype=torch.float32, device=DEVICE)
        else:
            obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)
            embed_tensor = None

        return (
            torch.tensor(np.array(obs_b), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(act_b), dtype=torch.long, device=DEVICE),
            torch.tensor(np.array(rew_b), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(nobs_b), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(done_b), dtype=torch.float32, device=DEVICE),
            embed_tensor,
        )

    def __len__(self): return len(self._buf)


class DQNAgent:
    """
    Double Dueling DQN Agent sử dụng PyTorch.
    Hỗ trợ optional analysis embedding (phân tích cơ bản/kỹ thuật từ LLM).
    """
    def __init__(
        self,
        obs_size:   int,
        n_actions:  int   = 3,
        hidden:     list | None  = None,
        lr:         float = 5e-4,
        lr_decay:   float = 0.995,
        lr_min:     float = 5e-5,
        gamma:      float = 0.97,
        tau:        float = 0.005,
        eps:        float = 1.0,
        eps_end:    float = 0.03,
        eps_decay:  float = 0.997,
        buffer_cap: int   = 20_000,
        batch_size: int   = 256,
        warmup:     int   = 1000,
        # ── Analysis embedding config ─────────────────────────────
        analysis_embed_dim: int | None = None,
        analysis_proj_layers: list[int] | None = None,
    ):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.tau        = tau
        self.eps        = eps
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.batch_size = batch_size
        self.warmup     = warmup

        self.lr         = lr
        self.lr_decay   = lr_decay
        self.lr_min     = lr_min
        self.current_lr = lr

        self.has_analysis = analysis_embed_dim is not None

        h = hidden or [256, 128, 64]
        self.q      = DuelingQNet(obs_size, n_actions, h,
                                  analysis_embed_dim=analysis_embed_dim,
                                  analysis_proj_layers=analysis_proj_layers).to(DEVICE)
        self.q_tgt  = DuelingQNet(obs_size, n_actions, h,
                                  analysis_embed_dim=analysis_embed_dim,
                                  analysis_proj_layers=analysis_proj_layers).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr, weight_decay=1e-4)
        self.buf    = ReplayBuffer(buffer_cap, has_analysis=self.has_analysis)

        self.steps  = 0
        self.learn_count = 0
        self.losses = deque(maxlen=10_000)
        self.episode_num = 0

        log.debug(f"[Agent] PyTorch initialized on {DEVICE}")
        if self.has_analysis:
            log.debug(f"[Agent] Analysis embedding enabled: dim={analysis_embed_dim}, proj={analysis_proj_layers}")

    def act(self, obs: np.ndarray, valid_actions: list[int] | None = None,
            greedy: bool = False, analysis_embed: np.ndarray | None = None) -> int:
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        if not greedy and np.random.rand() < self.eps:
            return int(np.random.choice(valid_actions))

        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        embed_t = None
        if self.has_analysis and analysis_embed is not None:
            embed_t = torch.tensor(analysis_embed, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q(obs_t, analysis_embed=embed_t).cpu().numpy()[0]

        # Action Masking
        mask = np.ones(self.n_actions, dtype=bool)
        mask[valid_actions] = False
        q_masked = q_values.copy()
        q_masked[mask] = -np.inf

        return int(np.argmax(q_masked))

    def store(self, obs, action, reward, next_obs, done, analysis_embed=None):
        self.buf.push(obs, action, reward, next_obs, done, analysis_embed=analysis_embed)
        self.steps += 1

    def learn(self) -> float | None:
        if len(self.buf) < self.warmup:
            return None

        obs_b, act_b, rew_b, nobs_b, done_b, embed_b = self.buf.sample(self.batch_size)

        # ── Double DQN target ───────────────────────────────────
        with torch.no_grad():
            best_a = self.q(nobs_b, analysis_embed=embed_b).argmax(dim=1, keepdim=True)
            q_next = self.q_tgt(nobs_b, analysis_embed=embed_b).gather(1, best_a).squeeze(1)
            target = rew_b + self.gamma * q_next * (1 - done_b)

        # ── Current Q-values ────────────────────────────────────
        q_cur = self.q(obs_b, analysis_embed=embed_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

        # ── Loss & Optimizer ────────────────────────────────────
        loss = F.huber_loss(q_cur, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft target update
        self._soft_update()

        loss_val = loss.item()
        self.losses.append(loss_val)
        self.learn_count += 1
        return loss_val

    def _soft_update(self):
        """θ_tgt = τ*θ + (1-τ)*θ_tgt"""
        for t_param, o_param in zip(self.q_tgt.parameters(), self.q.parameters()):
            t_param.data.copy_(self.tau * o_param.data + (1.0 - self.tau) * t_param.data)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def decay_lr(self):
        self.episode_num += 1
        new_lr = max(self.lr_min, self.lr * (self.lr_decay ** self.episode_num))
        self.current_lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "model_state": self.q.state_dict(),
            "target_state": self.q_tgt.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "eps": self.eps,
            "steps": self.steps,
            "learn_count": self.learn_count,
            "episode_num": self.episode_num,
            "current_lr": self.current_lr,
            "has_analysis": self.has_analysis,
        }
        torch.save(data, path)
        log.debug(f"[Agent] Saved → {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=DEVICE)
        self.q.load_state_dict(checkpoint["model_state"])
        self.q_tgt.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.eps = checkpoint.get("eps", self.eps_end)
        self.steps = checkpoint.get("steps", 0)
        self.learn_count = checkpoint.get("learn_count", 0)
        self.episode_num = checkpoint.get("episode_num", 0)
        self.current_lr = checkpoint.get("current_lr", self.lr)
        log.info(f"[Agent] Loaded ← {path} (ep={self.episode_num})")
