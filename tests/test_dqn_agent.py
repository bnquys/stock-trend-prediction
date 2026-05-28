"""
tests/test_dqn_agent.py
═══════════════════════════════════════════════════════════════════
Unit tests cho DQNAgent (PyTorch) — bao gồm:
  - Initialization
  - Forward pass shape
  - Action masking
  - Learning step (loss calculation)
  - Epsilon & LR decay
  - Model saving & loading
  - Buffer pushing & sampling
═══════════════════════════════════════════════════════════════════
"""
import sys, os, torch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl.agent.dqn_agent import DQNAgent, DEVICE

# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """Tạo agent cho testing."""
    return DQNAgent(
        obs_size=100,
        n_actions=3,
        hidden=[64, 32],
        lr=1e-3,
        warmup=10,
        batch_size=4
    )

# ─── Test Initialization ─────────────────────────────────────────────────

class TestAgentInit:
    def test_init_params(self, agent):
        assert agent.n_actions == 3
        assert agent.eps == 1.0
        assert agent.current_lr == 1e-3
        assert agent.steps == 0

    def test_networks_on_device(self, agent):
        assert next(agent.q.parameters()).device.type in [DEVICE.type, 'cpu']
        assert next(agent.q_tgt.parameters()).device.type in [DEVICE.type, 'cpu']

# ─── Test Acting ─────────────────────────────────────────────────────────

class TestAgentAct:
    def test_act_shape(self, agent):
        obs = np.random.randn(100).astype(np.float32)
        action = agent.act(obs, greedy=True)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_act_masking(self, agent):
        obs = np.random.randn(100).astype(np.float32)
        # Chỉ cho phép action 2
        action = agent.act(obs, valid_actions=[2], greedy=True)
        assert action == 2

    def test_random_act(self, agent):
        agent.eps = 1.0 # Luôn random
        obs = np.random.randn(100).astype(np.float32)
        actions = [agent.act(obs, greedy=False) for _ in range(50)]
        assert len(set(actions)) > 1 # Phải có tính ngẫu nhiên

# ─── Test Learning ───────────────────────────────────────────────────────

class TestAgentLearn:
    def test_store_and_sample(self, agent):
        obs = np.random.randn(100).astype(np.float32)
        agent.store(obs, 1, 1.0, obs, False)
        assert len(agent.buf) == 1
        assert agent.steps == 1

    def test_learn_warmup_guard(self, agent):
        # Buffer mới có 1, warmup=10 -> learn trả về None
        obs = np.random.randn(100).astype(np.float32)
        agent.store(obs, 1, 1.0, obs, False)
        loss = agent.learn()
        assert loss is None

    def test_learn_step(self, agent):
        obs = np.random.randn(100).astype(np.float32)
        for _ in range(15):
            agent.store(obs, 1, 1.0, obs, False)
        
        loss = agent.learn()
        assert isinstance(loss, float)
        assert loss >= 0
        assert agent.learn_count == 1

# ─── Test Decay ──────────────────────────────────────────────────────────

class TestAgentDecay:
    def test_epsilon_decay(self, agent):
        initial_eps = agent.eps
        agent.decay_epsilon()
        assert agent.eps < initial_eps
        assert agent.eps >= agent.eps_end

    def test_lr_decay(self, agent):
        initial_lr = agent.current_lr
        agent.decay_lr()
        assert agent.current_lr < initial_lr
        assert agent.current_lr >= agent.lr_min
        assert agent.episode_num == 1

# ─── Test Save & Load ────────────────────────────────────────────────────

class TestAgentSaveLoad:
    def test_save_load(self, agent, tmp_path):
        save_path = str(tmp_path / "test_model.pkl")
        agent.eps = 0.5
        agent.steps = 123
        agent.save(save_path)
        
        # Tạo agent mới với cùng architecture và load
        new_agent = DQNAgent(obs_size=100, n_actions=3, hidden=[64, 32])
        new_agent.load(save_path)
        
        assert new_agent.eps == 0.5
        assert new_agent.steps == 123
        
        # Test state dict weights match (xấp xỉ)
        for p1, p2 in zip(agent.q.parameters(), new_agent.q.parameters()):
            assert torch.equal(p1, p2)
