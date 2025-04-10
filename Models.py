"""
Models.py

このファイルは、ニューラルネットワーク関連のモデルやデータ保持、ノイズ生成のためのクラス群を定義します。
具体的には、以下の要素を含みます：
  - Actor：行動（ポリシー）を生成するネットワーク
  - Critic：状態・行動ペアの評価（Q値）を計算するネットワーク
  - ReplayBuffer：経験を保存し、バッチサンプルを行うためのバッファ
  - OrnsteinUhlenbeckProcess：探索用ノイズを生成するプロセス

※ DEVICE（計算に用いるデバイス）および各種ハイパーパラメータは Config.py および Utils.py からインポートしています。
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from Config import MEMORY_SIZE, BATCH_SIZE, MADDPG_HIDDEN_SIZE, CRITIC_INIT_W
from Utils import DEVICE  # DEVICE設定は Utils.py に定義

# --------------------------
# Replay Buffer
# --------------------------
class ReplayBuffer:
    def __init__(self, memory_size=MEMORY_SIZE):
        self.memory = deque(maxlen=memory_size)
     
    def cache(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))
     
    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))
        # GPU上に配置するため、DEVICEを指定してtensorに変換
        return (torch.tensor(state, dtype=torch.float32, device=DEVICE),
                torch.tensor(next_state, dtype=torch.float32, device=DEVICE),
                torch.tensor(action, dtype=torch.float32, device=DEVICE),
                torch.tensor(reward, dtype=torch.float32, device=DEVICE),
                torch.tensor(done, dtype=torch.float32, device=DEVICE))

# --------------------------
# Actor (Policy Network)
# --------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=MADDPG_HIDDEN_SIZE):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # softplusで非負出力し、5を上限とする
        x = torch.clamp(F.softplus(self.fc3(x)), 0, 5)
        return x

# --------------------------
# Critic (Q Network)
# --------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_size=MADDPG_HIDDEN_SIZE, init_w=CRITIC_INIT_W):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim * num_agents, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim * num_agents, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, states, actions):
        batch_size = states.size(0)
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        x = F.relu(self.fc1(states_flat))
        x = torch.cat([x, actions_flat], dim=1)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

# --------------------------
# Ornstein-Uhlenbeck Noise
# --------------------------
class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        self.x_prev = np.zeros(self.size)
    
    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x
