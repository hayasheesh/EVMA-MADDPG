"""
Agent.py

このファイルは、MADDPG エージェントクラスを定義します。
エージェントは、Actor と Critic のネットワーク、Replay Buffer、探索ノイズ生成などの要素を統合し、
行動の取得、経験の保存、ネットワーク更新、ターゲットネットワークのソフトアップデートなどの学習ロジックを実装します。
"""

import copy
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from Config import STATE_DIM, ACTION_DIM, NUM_AGENTS, MADDPG_HIDDEN_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, BATCH_SIZE, MEMORY_SIZE
from Models import Actor, Critic, ReplayBuffer, OrnsteinUhlenbeckProcess
from Utils import DEVICE as device

class MADDPG:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, num_agents=NUM_AGENTS, 
                 hidden_size=MADDPG_HIDDEN_SIZE, gamma=GAMMA, tau=TAU, lr_actor=LR_ACTOR, 
                 lr_critic=LR_CRITIC, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE,
                 use_transformer=False):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_transformer = use_transformer
        
        # Actor, Critic ネットワークの初期化とターゲットネットワークの作成
        if use_transformer:
            # Transformerベースのネットワークを使用
            from Models import TransformerActor, TransformerCritic
            self.actors = [TransformerActor(state_dim, action_dim, hidden_size).to(device) for _ in range(num_agents)]
            self.target_actors = [copy.deepcopy(actor).to(device) for actor in self.actors]
            self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
            
            self.critics = [TransformerCritic(state_dim, action_dim, num_agents, hidden_size).to(device) for _ in range(num_agents)]
            self.target_critics = [copy.deepcopy(critic).to(device) for critic in self.critics]
            self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.critics]
        else:
            # 通常のMADDPGネットワークを使用
            self.actors = [Actor(state_dim, action_dim, hidden_size).to(device) for _ in range(num_agents)]
            self.target_actors = [copy.deepcopy(actor).to(device) for actor in self.actors]
            self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
            
            self.critics = [Critic(state_dim, action_dim, num_agents, hidden_size).to(device) for _ in range(num_agents)]
            self.target_critics = [copy.deepcopy(critic).to(device) for critic in self.critics]
            self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.critics]
        
        self.buffer = ReplayBuffer(memory_size)
        self.loss_fn = torch.nn.MSELoss()
        self.noise = [OrnsteinUhlenbeckProcess(size=(action_dim,)) for _ in range(num_agents)]
    
    def get_actions(self, state, add_noise=True, decision_mode=False):
        actions = []
        for i in range(self.num_agents):
            # state[i] をテンソルに変換してネットワークに入力
            state_tensor = torch.tensor(state[i], dtype=torch.float32, device=device).unsqueeze(0)
            a = self.actors[i](state_tensor)
            a = a.detach().cpu().numpy().squeeze()
            if add_noise:
                noise_val = self.noise[i].sample()
                if np.ndim(noise_val) > 0:
                    noise_val = noise_val[0]
                a += noise_val
            # 行動を0〜5の範囲に制限
            a = np.clip(a, 0, 5)
            actions.append([a])
        return np.array(actions)
            
    def update(self):
        if len(self.buffer.memory) < self.batch_size:
            return
        
        state, next_state, action, reward, done = self.buffer.sample(self.batch_size)
        state_aug = state  # 状態拡張処理は不要なのでそのまま利用
        next_state_aug = next_state
        
        for agent in range(self.num_agents):
            target_actions = []
            for i in range(self.num_agents):
                a = self.target_actors[i](next_state_aug[:, i, :])
                target_actions.append(a)
            target_actions = torch.stack(target_actions, dim=1)
            
            target_q = self.target_critics[agent](next_state_aug, target_actions)
            r = reward[:, agent].unsqueeze(1)
            d = done[:, agent].unsqueeze(1)
            td_target = r + self.gamma * target_q * (1 - d)
            
            current_q = self.critics[agent](state_aug, action)
            critic_loss = self.loss_fn(current_q, td_target.detach())
            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critics[agent].parameters(), 0.5)
            self.critic_optimizers[agent].step()
            
            current_actions = []
            for i in range(self.num_agents):
                if i == agent:
                    current_actions.append(self.actors[i](state_aug[:, i, :]))
                else:
                    current_actions.append(self.actors[i](state_aug[:, i, :]).detach())
            current_actions = torch.stack(current_actions, dim=1)
            actor_loss = -self.critics[agent](state_aug, current_actions).mean()
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actors[agent].parameters(), 0.5)
            self.actor_optimizers[agent].step()
            
            self.soft_update(self.actors[agent], self.target_actors[agent])
            self.soft_update(self.critics[agent], self.target_critics[agent])
    
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
