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
# Transformer-based Actor
# --------------------------
class TransformerActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=MADDPG_HIDDEN_SIZE, num_heads=4, num_layers=2):
        super(TransformerActor, self).__init__()
        self.state_embedding = nn.Linear(state_dim, hidden_size)
        
        # Transformerエンコーダ層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 出力層
        self.output_layer = nn.Linear(hidden_size, action_dim)
    
    def forward(self, state):
        # 状態を埋め込み
        x = self.state_embedding(state)
        
        # バッチサイズを取得
        batch_size = x.size(0)
        
        # シーケンスとして扱うために形状を変更
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Transformerエンコーダを適用
        x = self.transformer_encoder(x)
        
        # 出力層を適用
        x = self.output_layer(x.squeeze(1))
        
        # 出力を0-5の範囲にスケーリング
        x = torch.sigmoid(x) * 5
        
        return x

# --------------------------
# Transformer-based Critic
# --------------------------
class TransformerCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_size=MADDPG_HIDDEN_SIZE, num_heads=4, num_layers=2, init_w=CRITIC_INIT_W):
        super(TransformerCritic, self).__init__()
        self.state_embedding = nn.Linear(state_dim * num_agents, hidden_size)
        self.action_embedding = nn.Linear(action_dim * num_agents, hidden_size)
        
        # Transformerエンコーダ層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 出力層
        self.output_layer = nn.Linear(hidden_size, 1)
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, states, actions):
        batch_size = states.size(0)
        
        # 状態と行動を埋め込み
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        
        state_emb = self.state_embedding(states_flat)
        action_emb = self.action_embedding(actions_flat)
        
        # 状態と行動の埋め込みを結合
        x = state_emb + action_emb
        
        # シーケンスとして扱うために形状を変更
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Transformerエンコーダを適用
        x = self.transformer_encoder(x)
        
        # 出力層を適用
        q = self.output_layer(x.squeeze(1))
        
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
