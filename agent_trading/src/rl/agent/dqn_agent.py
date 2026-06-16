"""
src/rl/agent/dqn_agent.py
════════════════════════════════════════════════════════════════════════════
Double Dueling DQN Agent
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingQNet(nn.Module):
    """
    Dueling DQN Architecture with LSTM/GRU encoders for time-series.
    Input -> recurrent encoders -> Shared Layers -> (Value Stream & Advantage Stream) -> Q(s,a)
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: list[int], window: int = 20, port_size: int = 11):
        super().__init__()
        self.window = window
        self.port_size = port_size
        if (obs_size - self.port_size) % self.window != 0:
            self.port_size = obs_size % self.window
        self.n_features = (obs_size - self.port_size) // window
        
        # LSTM Feature Extraction
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        
        self.gru = nn.GRU(
            input_size=self.n_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        lstm_out_size = 128
        gru_out_size = 64
        
        # Shared Feature Extraction
        layers = []
        prev = lstm_out_size + gru_out_size + self.n_features + self.port_size
        for h in hidden[:-1]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(p=0.2)) # Thêm Dropout 20%
            prev = h
        self.feature_layer = nn.Sequential(*layers)
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Tách chuỗi thời gian và trạng thái danh mục
        if self.port_size > 0:
            seq = x[:, :-self.port_size]
            port = x[:, -self.port_size:]
        else:
            seq = x
            port = x.new_zeros((batch_size, 0))
        
        # Reshape sequence for recurrent trend encoders.
        seq_time = seq.view(batch_size, self.window, self.n_features)
        
        _, (lstm_hidden, _) = self.lstm(seq_time)
        lstm_out = lstm_hidden[-1]
        _, gru_hidden = self.gru(seq_time)
        gru_out = gru_hidden[-1]
        
        # Use the most recent day's raw feature vector alongside recurrent encodings.
        current_day_features = seq_time[:, -1, :]
        
        features_in = torch.cat([lstm_out, gru_out, current_day_features, port], dim=1)
        features = self.feature_layer(features_in)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q


class ReplayBuffer:
    def __init__(self, cap: int = 20_000):
        self._buf: deque = deque(maxlen=cap)

    def push(self, obs, action, reward, next_obs, done, valid_actions, next_valid_actions):
        n_actions = 3
        # Valid mask
        mask = np.zeros(n_actions, dtype=np.bool_)
        mask[valid_actions] = True
        
        nmask = np.zeros(n_actions, dtype=np.bool_)
        nmask[next_valid_actions] = True
        
        self._buf.append((
            obs.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            float(done),
            mask,
            nmask
        ))

    def sample(self, bs: int):
        idx = np.random.choice(len(self._buf), bs, replace=False)
        batch = [self._buf[i] for i in idx]
        obs_b, act_b, rew_b, nobs_b, done_b, mask_b, nmask_b = zip(*batch)
        
        return (
            torch.tensor(np.array(obs_b), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(act_b), dtype=torch.long, device=DEVICE),
            torch.tensor(np.array(rew_b), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(nobs_b), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(done_b), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(mask_b), dtype=torch.bool, device=DEVICE),
            torch.tensor(np.array(nmask_b), dtype=torch.bool, device=DEVICE)
        )

    def __len__(self): return len(self._buf)


class DQNAgent:
    """
    Double Dueling DQN Agent.
    """
    def __init__(
        self,
        obs_size:   int,
        n_actions:  int   = 3,
        hidden:     list  = None,
        lr:         float = 5e-4,
        lr_decay:   float = 0.995,
        lr_min:     float = 5e-5,
        gamma:      float = 0.97,
        tau:        float = 0.005,
        eps:        float = 1.0,
        eps_end:    float = 0.08,
        eps_decay:  float = 0.997,
        buffer_cap: int   = 50_000,
        batch_size: int   = 256,
        warmup:     int   = 1000,
        window:     int   = 20,
        port_size:  int   = 11,
        learn_every:int   = 1,
        weight_decay:float= 1e-5,
    ):
        self.n_actions  = n_actions
        self.obs_size   = obs_size
        self.hidden     = hidden or [256, 128, 64]
        self.window     = window
        self.port_size  = port_size
        self.gamma      = gamma
        self.tau        = tau
        self.eps        = eps
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.batch_size = batch_size
        self.warmup     = warmup
        self.learn_every= learn_every
        
        self.lr         = lr
        self.lr_decay   = lr_decay
        self.lr_min     = lr_min
        self.current_lr = lr

        h = self.hidden
        self.q      = DuelingQNet(obs_size, n_actions, h, window, port_size).to(DEVICE)
        self.q_tgt  = DuelingQNet(obs_size, n_actions, h, window, port_size).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval() # Target net luôn ở mode eval

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)
        self.buf    = ReplayBuffer(buffer_cap)
        
        self.steps  = 0
        self.learn_count = 0
        self.losses = deque(maxlen=10_000)
        self.episode_num = 0

        log.info(f"[Agent] PyTorch initialized on {DEVICE}")

    def act(self, obs: np.ndarray, valid_actions: list[int] | None = None, greedy: bool = False) -> int:
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
           
        if not greedy and np.random.rand() < self.eps:
            return int(np.random.choice(valid_actions))
            
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q(obs_t).cpu().numpy()[0]

        # Noisy exploration: khi epsilon nhỏ, thêm noise nhẹ vào Q-values
        # để duy trì khám phá và tránh policy collapse
        if not greedy and self.eps < 0.15:
            noise_scale = 0.01 * (0.15 - self.eps) / 0.15  # Noise tỷ lệ nghịch với eps
            noise = np.random.normal(0, noise_scale, size=q_values.shape)
            q_values = q_values + noise

        # Action Masking
        mask = np.ones(self.n_actions, dtype=bool)
        mask[valid_actions] = False
        q_masked = q_values.copy()
        q_masked[mask] = -np.inf
        
        return int(np.argmax(q_masked))

    def store(self, obs, action, reward, next_obs, done, valid_actions=None, next_valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        if next_valid_actions is None:
            next_valid_actions = list(range(self.n_actions))
        self.buf.push(obs, action, reward, next_obs, done, valid_actions, next_valid_actions)
        self.steps += 1

    def learn(self) -> float | None:
        if len(self.buf) < self.warmup:
            return None
            
        if self.steps % self.learn_every != 0:
            return None

        obs_b, act_b, rew_b, nobs_b, done_b, mask_b, nmask_b = self.buf.sample(self.batch_size)

        # ── Double DQN target with Action Masking ───────────────────
        with torch.no_grad():
            # Đánh giá Q-values cho next state từ online net
            q_next_online = self.q(nobs_b)
            # Áp dụng Action Masking: Các action không hợp lệ sẽ bị phạt -inf
            q_next_online[~nmask_b] = float('-inf')
            
            # Online net chọn action cho next state
            best_a = q_next_online.argmax(dim=1, keepdim=True)
            
            # Target net đánh giá action đó
            q_next = self.q_tgt(nobs_b).gather(1, best_a).squeeze(1)
            target = rew_b + self.gamma * q_next * (1 - done_b)

        # ── Current Q-values ────────────────────────────────────
        q_cur = self.q(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

        # ── Loss & Optimizer ────────────────────────────────────
        loss = F.huber_loss(q_cur, target) # Huber loss robust hơn MSE

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft target update
        self._soft_update()

        loss_val = loss.item()
        self.losses.append(loss_val)
        self.learn_count += 1
        return loss_val

    def _soft_update(self):
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
            "obs_size": self.obs_size,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "window": self.window,
            "port_size": self.port_size,
            "lr": self.lr,
            "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0.0),
        }
        torch.save(data, path)
        log.info(f"[Agent] Saved PyTorch model → {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=DEVICE)
        ckpt_hidden = checkpoint.get("hidden")
        ckpt_obs_size = checkpoint.get("obs_size")
        ckpt_window = checkpoint.get("window", self.window)
        ckpt_port_size = checkpoint.get("port_size", self.port_size)
        if ckpt_hidden and ckpt_obs_size and (
            ckpt_hidden != self.hidden
            or ckpt_obs_size != self.obs_size
            or ckpt_window != self.window
            or ckpt_port_size != self.port_size
        ):
            self.obs_size = ckpt_obs_size
            self.hidden = ckpt_hidden
            self.window = ckpt_window
            self.port_size = ckpt_port_size
            self.q = DuelingQNet(self.obs_size, self.n_actions, self.hidden, self.window, self.port_size).to(DEVICE)
            self.q_tgt = DuelingQNet(self.obs_size, self.n_actions, self.hidden, self.window, self.port_size).to(DEVICE)
            self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr, weight_decay=checkpoint.get("weight_decay", 0.0))
        self.q.load_state_dict(checkpoint["model_state"])
        self.q_tgt.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.eps = checkpoint.get("eps", self.eps_end)
        self.steps = checkpoint.get("steps", 0)
        self.learn_count = checkpoint.get("learn_count", 0)
        self.episode_num = checkpoint.get("episode_num", 0)
        self.current_lr = checkpoint.get("current_lr", self.lr)
        log.info(f"[Agent] Loaded PyTorch model ← {path} (ep={self.episode_num})")
