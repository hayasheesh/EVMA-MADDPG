"""
EVEnv.py

このファイルは、EV（電気自動車）環境クラスを定義します。
各エージェントの充電状態（SoC）、報酬計算、ステップごとの状態更新など、シミュレーションのロジックを実装しています。

主な機能：
  - reset() : 環境状態の初期化
  - get_state_for_agent() : エージェントごとに状態ベクトル（自身のSoC、相手のSoC、エージェント識別子、AG充電要請、相手の充電予測値または実測値）を生成
  - step_sequential() : 逐次実行モードでの行動処理（先行エージェントと後攻エージェントの順次処理）
"""

import random
import numpy as np
from Config import (
    EPISODE_STEPS, INITIAL_SOC, CAPACITY, AG_REQUEST, PENALTY_WEIGHT, 
    TOLERANCE_NARROW, TOLERANCE_WIDE
)

class EVEnv:
    def __init__(self, capacity=CAPACITY, initial_soc=INITIAL_SOC, ag_request=AG_REQUEST, 
                episode_steps=EPISODE_STEPS):
        self.capacity = capacity  
        self.initial_soc = initial_soc  
        # Configで設定したAG_REQUESTを使用
        self.ag_request = ag_request
        self.episode_steps = episode_steps  
        self.reset()
    
    def reset(self):
        self.soc = {"ev1": random.uniform(0, 100), "ev2": random.uniform(0, 100)}
        self.initial_soc = self.soc.copy()  # 各EVの初期SoC値を保存
        
        # 両エージェントの充電可能量を計算
        total_available_capacity = (2 * self.capacity) - (self.soc["ev1"] + self.soc["ev2"])
        
        # dirichlet分布を使って48ステップ分のAG要請を生成
        # alpha値を全て1にすると一様なdirichlet分布になる
        alpha = np.ones(self.episode_steps)
        ag_request_ratios = np.random.dirichlet(alpha)
        
        # 合計が total_available_capacity になるようにスケール
        self.ag_requests_for_episode = ag_request_ratios * total_available_capacity
        
        # 各ステップの要請値が10を超えないように制限
        # 10を超える要請値がある場合は、超過分を他のステップに再分配
        while np.any(self.ag_requests_for_episode > 10.0):
            # 10を超える要請値を特定
            over_limit_indices = np.where(self.ag_requests_for_episode > 10.0)[0]
            under_limit_indices = np.where(self.ag_requests_for_episode < 10.0)[0]
            
            if len(under_limit_indices) == 0:
                # すべてのステップが10に近い場合、均等に分配
                self.ag_requests_for_episode = np.ones(self.episode_steps) * (total_available_capacity / self.episode_steps)
                break
            
            for idx in over_limit_indices:
                excess = self.ag_requests_for_episode[idx] - 10.0
                self.ag_requests_for_episode[idx] = 10.0
                
                # 余剰分を10未満のステップに分配
                distribution_weights = 10.0 - self.ag_requests_for_episode[under_limit_indices]
                if np.sum(distribution_weights) > 0:
                    distribution_ratios = distribution_weights / np.sum(distribution_weights)
                    redistribution = excess * distribution_ratios
                    self.ag_requests_for_episode[under_limit_indices] += redistribution
        
        self.step_count = 0
        self.temp_actions = {}
        self.temp_predicted_first = None
        self.temp_predicted_second = None
        self.temp_first_agent = None
        
        # 最初のステップのAG要請をセット
        self.ag_request = self.ag_requests_for_episode[0]
        
        state_ev1 = self.get_state_for_agent("ev1", other_charge=None)
        state_ev2 = self.get_state_for_agent("ev2", other_charge=None)
        return np.array([state_ev1, state_ev2])

    
    def get_state_for_agent(self, agent, other_charge=None):
        # other_chargeがNoneの場合は予測値（なければ0～5の乱数）を利用
        if other_charge is None:
            if agent == "ev1":
                other_val = self.temp_predicted_second if self.temp_predicted_second is not None else random.uniform(0, 5)
            elif agent == "ev2":
                other_val = self.temp_predicted_first if self.temp_predicted_first is not None else random.uniform(0, 5)
            else:
                raise ValueError("Unknown agent")
        else:
            other_val = other_charge

        if agent == "ev1":
            return np.array([self.soc["ev1"], self.soc["ev2"], self.ag_request, other_val], dtype=np.float32)
        elif agent == "ev2":
            return np.array([self.soc["ev2"], self.soc["ev1"], self.ag_request, other_val], dtype=np.float32)
        else:
            raise ValueError("Unknown agent")
        
    def get_first_agent(self):
        # ステップカウントにより先行エージェントを交互に決定
        return "ev1" if ((self.step_count + 1) % 2) == 1 else "ev2"
    
    def step_sequential(self, agent_id, action):
        """
        逐次実行モードでのステップ処理を行います。
        先行エージェントの行動を受け取り、次に後攻エージェントの行動を受け付け、
        両エージェントの行動結果から状態更新と報酬計算を行います。
        """
        if not self.temp_actions:
            first_agent = self.get_first_agent()
            if agent_id != first_agent:
                raise ValueError(f"Expected first action from {first_agent}, but got {agent_id}")
            state = self.get_state_for_agent(agent_id, other_charge=self.temp_predicted_first)
            self.temp_actions[agent_id] = action
            self.temp_first_agent = agent_id
            return state, None, False, {"message": "Waiting for second agent action"}
        else:
            expected_second = "ev2" if self.temp_first_agent == "ev1" else "ev1"
            if agent_id != expected_second:
                raise ValueError(f"Expected second action from {expected_second}, but got {agent_id}")
            self.temp_actions[agent_id] = action
            a1 = self.temp_actions.get("ev1", 0.0)
            a2 = self.temp_actions.get("ev2", 0.0)
            if self.temp_first_agent == "ev1":
                true_first_soc = self.soc["ev1"]
            else:
                true_first_soc = self.soc["ev2"]
            
            # 逐次実行時の充電更新（CAPACITYを上限とする）
            penalty_ev1 = False
            if self.soc["ev1"] + a1 > self.capacity:
                self.soc["ev1"] = self.capacity
                penalty_ev1 = True
            else:
                self.soc["ev1"] += a1
            
            penalty_ev2 = False
            if self.soc["ev2"] + a2 > self.capacity:
                self.soc["ev2"] = self.capacity
                penalty_ev2 = True
            else:
                self.soc["ev2"] += a2
            
            total_charge = a1 + a2
            deviation = abs(total_charge - self.ag_request)
            if deviation <= TOLERANCE_NARROW:
                reward_total = 100
            elif deviation < TOLERANCE_WIDE:
                reward_total = 20 * (TOLERANCE_WIDE - deviation)
            else:
                reward_total = 0
            
            reward_ev1 = reward_total / 2.0
            reward_ev2 = reward_total / 2.0
            if penalty_ev1:
                reward_ev1 -= PENALTY_WEIGHT
            if penalty_ev2:
                reward_ev2 -= PENALTY_WEIGHT

            self.step_count += 1
            
            # 次ステップ用にAG要請を更新（事前生成したリストから取得）
            if self.step_count < self.episode_steps:
                self.ag_request = self.ag_requests_for_episode[self.step_count]
            
            state_ev1 = self.get_state_for_agent("ev1", other_charge=a2)
            state_ev2 = self.get_state_for_agent("ev2", other_charge=a1)
            next_state = np.array([state_ev1, state_ev2])
            rewards = np.array([reward_ev1, reward_ev2], dtype=np.float32)
            done_flags = np.array([self.step_count >= self.episode_steps, self.step_count >= self.episode_steps], dtype=np.float32)
            info = {
                "total_charge": total_charge,
                "reward_total": reward_total,
            }
            self.temp_actions = {}
            self.temp_predicted_first = None
            self.temp_predicted_second = None
            self.temp_first_agent = None
            return next_state, rewards, done_flags, info
