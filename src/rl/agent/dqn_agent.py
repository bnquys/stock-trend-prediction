"""
src/rl/agent/dqn_agent.py
════════════════════════════════════════════════════════════════════════════
Double Dueling DQN Agent — PyTorch Implementation
Hỗ trợ Analysis Embedding (phân tích cơ bản) qua Addition fusion.
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import pickle, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

log = logging.getLogger(__name__)

# Tự động chọn Device (GPU nếu có)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU optimization: ensure PyTorch uses all available cores
if DEVICE.type == "cpu":
    import os
    _num_threads = int(os.environ.get("TORCH_NUM_THREADS", 0)) or os.cpu_count() or 4
    torch.set_num_threads(_num_threads)
    torch.set_num_interop_threads(max(1, _num_threads // 2))


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
    """
    Numpy-backed Replay Buffer — tối ưu cho CPU training.
    Pre-allocate numpy arrays, tránh tạo list/tuple mỗi step.
    Sample trực tiếp từ numpy → torch (1 copy thay vì N copies).
    """
    def __init__(self, cap: int = 20_000, obs_size: int = 0,
                 has_analysis: bool = False, embed_dim: int = 0):
        self._cap = cap
        self._size = 0
        self._ptr = 0
        self.has_analysis = has_analysis

        # Pre-allocate numpy arrays
        self._obs = np.zeros((cap, obs_size), dtype=np.float32)
        self._next_obs = np.zeros((cap, obs_size), dtype=np.float32)
        self._actions = np.zeros(cap, dtype=np.int64)
        self._rewards = np.zeros(cap, dtype=np.float32)
        self._dones = np.zeros(cap, dtype=np.float32)

        if has_analysis and embed_dim > 0:
            self._embeds = np.zeros((cap, embed_dim), dtype=np.float32)
            self._embed_mask = np.zeros(cap, dtype=np.bool_)
        else:
            self._embeds = None
            self._embed_mask = None

    def push(self, obs, action, reward, next_obs, done, analysis_embed=None):
        i = self._ptr
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)

        if self.has_analysis and self._embeds is not None:
            if analysis_embed is not None:
                self._embeds[i] = analysis_embed
                self._embed_mask[i] = True
            else:
                self._embed_mask[i] = False

        self._ptr = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, bs: int):
        idx = np.random.randint(0, self._size, size=bs)

        obs_t = torch.from_numpy(self._obs[idx]).to(DEVICE)
        act_t = torch.from_numpy(self._actions[idx]).to(DEVICE)
        rew_t = torch.from_numpy(self._rewards[idx]).to(DEVICE)
        nobs_t = torch.from_numpy(self._next_obs[idx]).to(DEVICE)
        done_t = torch.from_numpy(self._dones[idx]).to(DEVICE)

        embed_t = None
        if self.has_analysis and self._embeds is not None:
            if self._embed_mask[idx].any():
                embed_t = torch.from_numpy(self._embeds[idx]).to(DEVICE)

        return obs_t, act_t, rew_t, nobs_t, done_t, embed_t

    def __len__(self): return self._size


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
        # Disable gradient tracking for target network — saves CPU overhead
        for p in self.q_tgt.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr, weight_decay=1e-4)
        self.buf    = ReplayBuffer(
            cap=buffer_cap,
            obs_size=obs_size,
            has_analysis=self.has_analysis,
            embed_dim=analysis_embed_dim or 0,
        )

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

        # as_tensor avoids copy for contiguous float32 numpy arrays on CPU
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        embed_t = None
        if self.has_analysis and analysis_embed is not None:
            embed_t = torch.as_tensor(analysis_embed, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.inference_mode():
            q_values = self.q(obs_t, analysis_embed=embed_t).squeeze(0)

        # Action Masking — direct numpy for small valid_actions (faster than tensor on CPU)
        q_np = q_values.numpy()
        best_idx = 0
        best_val = -float('inf')
        for i, a in enumerate(valid_actions):
            if q_np[a] > best_val:
                best_val = q_np[a]
                best_idx = i
        return valid_actions[best_idx]

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

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft target update
        self._soft_update()

        loss_val = loss.item()
        self.losses.append(loss_val)
        self.learn_count += 1
        return loss_val

    @torch.no_grad()
    def _soft_update(self):
        """θ_tgt = τ*θ + (1-τ)*θ_tgt — in-place ops to avoid temp tensors."""
        for t_param, o_param in zip(self.q_tgt.parameters(), self.q.parameters()):
            t_param.data.mul_(1.0 - self.tau).add_(o_param.data, alpha=self.tau)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def decay_lr(self):
        self.episode_num += 1
        new_lr = max(self.lr_min, self.lr * (self.lr_decay ** self.episode_num))
        self.current_lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
