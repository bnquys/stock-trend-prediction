"""Quick benchmark for agent performance."""
import torch
import numpy as np
import time
from src.rl.agent.dqn_agent import DQNAgent, DEVICE

print(f"Device: {DEVICE}")
print(f"Threads: {torch.get_num_threads()}")
print(f"torch.compile available: {hasattr(torch, 'compile')}")

# Test agent
agent = DQNAgent(obs_size=147, n_actions=3, hidden=[256, 128, 64])

# Benchmark act() speed
obs = np.random.randn(147).astype(np.float32)
valid = [0, 1]

# Warmup (important for torch.compile)
for _ in range(20):
    agent.act(obs, valid_actions=valid, greedy=True)

# Benchmark act
t0 = time.perf_counter()
N = 1000
for _ in range(N):
    agent.act(obs, valid_actions=valid, greedy=True)
elapsed = time.perf_counter() - t0
print(f"act() x{N}: {elapsed*1000:.1f}ms ({elapsed/N*1000:.3f}ms/call)")

# Benchmark learn()
for i in range(1200):
    agent.store(obs, 0, 0.1, obs, False)

# Warmup learn
for _ in range(5):
    agent.learn()

t0 = time.perf_counter()
N2 = 100
for _ in range(N2):
    agent.learn()
elapsed2 = time.perf_counter() - t0
print(f"learn() x{N2}: {elapsed2*1000:.1f}ms ({elapsed2/N2*1000:.3f}ms/call)")

# Test save/load
agent.save("models/test_save.pkl")
agent.load("models/test_save.pkl")
print("save/load OK")
import os
os.remove("models/test_save.pkl")
print("ALL OK")
