"""
src/rl/agent/dqn_agent.py
════════════════════════════════════════════════════════════════════════════
DQN Agent — đơn giản, đúng, không có Dueling để tránh bug gradient.

Architecture: Linear (vanilla Q-net, đã test kỹ)
  Input → FC(128,ReLU) → FC(64,ReLU) → Q(3)

Features:
  - Double DQN (online chọn action, target evaluate)
  - Soft target update
  - ε-greedy decay per-episode
  - Global gradient norm clipping
  - Adam optimizer
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, pickle, logging
import numpy as np
from collections import deque

log = logging.getLogger(__name__)


# ─── Activation ──────────────────────────────────────────────────────────
def relu(x):   return np.maximum(0.0, x)
def relu_g(x): return (x > 0.0).astype(np.float64)


# ─── Linear Layer ────────────────────────────────────────────────────────
class FC:
    """Fully-connected layer với Adam optimizer."""

    def __init__(self, in_d: int, out_d: int, lr: float = 5e-4):
        # He initialization
        std = np.sqrt(2.0 / in_d)
        self.W  = (np.random.randn(in_d, out_d) * std).astype(np.float64)
        self.b  = np.zeros(out_d, np.float64)
        self.lr = lr
        # Adam
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t  = 0
        self._x = None  # cache for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x.astype(np.float64)
        return self._x @ self.W + self.b   # shape (B, out_d)

    def backward(self, g_out: np.ndarray) -> np.ndarray:
        """
        g_out: gradient w.r.t. output, shape (B, out_d)
        Returns: gradient w.r.t. input, shape (B, in_d)
        """
        g_out = g_out.astype(np.float64)
        B = g_out.shape[0]

        g_W  = self._x.T @ g_out / B       # (in_d, out_d)
        g_b  = g_out.mean(axis=0)           # (out_d,)
        g_in = g_out @ self.W.T             # (B, in_d)

        # Global grad norm clip
        norm = float(np.sqrt(np.sum(g_W**2) + np.sum(g_b**2))) + 1e-8
        if norm > 1.0:
            g_W /= norm
            g_b /= norm

        # Adam update
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.mW = b1 * self.mW + (1 - b1) * g_W
        self.vW = b2 * self.vW + (1 - b2) * g_W**2
        mWh = self.mW / (1 - b1**self.t)
        vWh = self.vW / (1 - b2**self.t)
        self.W -= self.lr * mWh / (np.sqrt(vWh) + eps)

        self.mb = b1 * self.mb + (1 - b1) * g_b
        self.vb = b2 * self.vb + (1 - b2) * g_b**2
        mbh = self.mb / (1 - b1**self.t)
        vbh = self.vb / (1 - b2**self.t)
        self.b -= self.lr * mbh / (np.sqrt(vbh) + eps)

        return g_in

    def copy_from(self, other: "FC") -> None:
        self.W = other.W.copy(); self.b = other.b.copy()

    def soft_update(self, other: "FC", tau: float) -> None:
        self.W = tau * other.W + (1 - tau) * self.W
        self.b = tau * other.b + (1 - tau) * self.b


# ─── Q-Network ───────────────────────────────────────────────────────
class QNet:
    """
    Vanilla Q-network: Input → hidden layers → Q(n_actions)
    Mỗi FC layer theo sau bởi ReLU, trừ layer cuối (linear output).
    """

    def __init__(self, obs_size: int, n_actions: int,
                 hidden: list[int], lr: float):
        sizes = [obs_size] + hidden + [n_actions]
        self.layers: list[FC] = [
            FC(sizes[i], sizes[i + 1], lr)
            for i in range(len(sizes) - 1)
        ]
        self.n_actions = n_actions
        self._pre: list[np.ndarray] = []   # pre-activation cache

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, obs_size) hoặc (obs_size,). Returns Q-values (B, n_actions)."""
        sq = (x.ndim == 1)
        if sq: x = x[np.newaxis, :]
        x = x.astype(np.float64)

        self._pre = []
        out = x
        for i, layer in enumerate(self.layers[:-1]):
            pre = layer.forward(out)
            self._pre.append(pre)
            out = relu(pre)
        # Last layer: linear
        q = self.layers[-1].forward(out)
        q = np.clip(q, -20.0, 20.0)   # clip để tránh explosion
        return q[0] if sq else q

    def backward(self, g_q: np.ndarray) -> None:
        """
        g_q: gradient w.r.t. Q-values, shape (B, n_actions).
        Backward qua tất cả layers.
        """
        g = np.clip(g_q.astype(np.float64), -2.0, 2.0)

        # Last layer backward
        g = self.layers[-1].backward(g)

        # Hidden layers backward (reversed)
        for i in reversed(range(len(self.layers) - 1)):
            g = g * relu_g(self._pre[i])   # backward qua ReLU
            g = self.layers[i].backward(g)

    def copy_from(self, other: "QNet") -> None:
        for t, s in zip(self.layers, other.layers): t.copy_from(s)

    def soft_update(self, other: "QNet", tau: float) -> None:
        for t, s in zip(self.layers, other.layers): t.soft_update(s, tau)


# ─── Replay Buffer ───────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, cap: int = 20_000):
        self._buf: deque = deque(maxlen=cap)

    def push(self, obs, action, reward, next_obs, done):
        self._buf.append((
            obs.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            float(done),
        ))

    def sample(self, bs: int):
        idx = np.random.choice(len(self._buf), bs, replace=False)
        batch = [self._buf[i] for i in idx]
        obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)
        return (np.array(obs_b,  dtype=np.float32),
                np.array(act_b,  dtype=np.int32),
                np.array(rew_b,  dtype=np.float32),
                np.array(nobs_b, dtype=np.float32),
                np.array(done_b, dtype=np.float32))

    def __len__(self): return len(self._buf)


# ─── DQN Agent ───────────────────────────────────────────────────────
class DQNAgent:
    """
    Double DQN (vanilla, không Dueling).
    Epsilon decay per-episode.
    """

    def __init__(
        self,
        obs_size:   int,
        n_actions:  int   = 3,
        hidden:     list  = None,
        lr:         float = 5e-4,
        gamma:      float = 0.95,
        tau:        float = 0.01,
        eps:        float = 1.0,
        eps_end:    float = 0.05,
        eps_decay:  float = 0.995,   # nhân eps sau mỗi episode
        buffer_cap: int   = 20_000,
        batch_size: int   = 64,
        warmup:     int   = 500,
    ):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.tau        = tau
        self.eps        = eps
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.batch_size = batch_size
        self.warmup     = warmup

        h = hidden or [128, 64]
        self.q      = QNet(obs_size, n_actions, h, lr)
        self.q_tgt  = QNet(obs_size, n_actions, h, lr)
        self.q_tgt.copy_from(self.q)

        self.buf    = ReplayBuffer(buffer_cap)
        self.steps  = 0        # tổng số bước train
        self.learn_count = 0   # tổng số lần backward
        self.losses: list[float] = []

    # ─── Act ─────────────────────────────────────────────────────
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        q = self.q.forward(obs)
        return int(np.argmax(q))

    # ─── Store & Learn ────────────────────────────────────────────
    def store(self, obs, action, reward, next_obs, done):
        self.buf.push(obs, action, reward, next_obs, done)
        self.steps += 1

    def learn(self) -> float | None:
        """
        Học 1 mini-batch. Trả về loss hoặc None nếu buffer chưa đủ.
        Gọi sau MỖI BƯỚC trong episode.
        """
        if len(self.buf) < self.warmup:
            return None

        obs_b, act_b, rew_b, nobs_b, done_b = self.buf.sample(self.batch_size)

        # ── Double DQN target ───────────────────────────────────
        # Online net chọn action tốt nhất cho next state
        q_next_online  = self.q.forward(nobs_b.astype(np.float64))    # (B, A)
        best_a         = np.argmax(q_next_online, axis=1)             # (B,)
        # Target net đánh giá action đó
        q_next_target  = self.q_tgt.forward(nobs_b.astype(np.float64))# (B, A)
        nv             = q_next_target[np.arange(self.batch_size), best_a]  # (B,)

        target = rew_b.astype(np.float64) + self.gamma * nv * (1 - done_b.astype(np.float64))
        target = np.clip(target, -50, 50)

        # ── Current Q-values ────────────────────────────────────
        q_cur = self.q.forward(obs_b.astype(np.float64))              # (B, A)
        cur   = q_cur[np.arange(self.batch_size), act_b]              # (B,)

        td    = cur - target                                           # (B,)
        loss  = float(np.mean(td**2))

        # ── Gradient ────────────────────────────────────────────
        # Huber: clip TD-error tránh explosion
        td_clip = np.clip(td, -5.0, 5.0)
        g_q     = np.zeros_like(q_cur)
        g_q[np.arange(self.batch_size), act_b] = td_clip / self.batch_size

        if np.isfinite(loss) and loss < 1e6:
            self.q.backward(g_q)
            self.losses.append(loss)
            self.learn_count += 1

        # Soft update target
        self.q_tgt.soft_update(self.q, self.tau)

        return loss

    def decay_epsilon(self):
        """Gọi sau mỗi episode."""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ─── Save / Load ──────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "q_layers":   [(l.W, l.b) for l in self.q.layers],
            "tgt_layers": [(l.W, l.b) for l in self.q_tgt.layers],
            "eps":        self.eps,
            "steps":      self.steps,
            "learn_count":self.learn_count,
            "losses":     self.losses[-10_000:],
        }
        with open(path, "wb") as f: pickle.dump(data, f)
        log.info(f"[Agent] Saved → {path}  (eps={self.eps:.4f})")

    def load(self, path: str) -> None:
        with open(path, "rb") as f: data = pickle.load(f)
        for l, (W, b) in zip(self.q.layers,     data["q_layers"]):   l.W=W; l.b=b
        for l, (W, b) in zip(self.q_tgt.layers, data["tgt_layers"]): l.W=W; l.b=b
        self.eps         = data.get("eps",    self.eps_end)
        self.steps       = data.get("steps",  0)
        self.learn_count = data.get("learn_count", 0)
        self.losses      = data.get("losses", [])
        log.info(f"[Agent] Loaded ← {path}  (eps={self.eps:.4f})")
