"""
EVEnv.py

このファイルは、EV（電気自動車）環境クラスを定義します。
各エージェントの充電状態（SoC）、報酬計算、ステップごとの状態更新など、シミュレーションのロジックを実装しています。

主な機能：
  - reset() : 環境状態の初期化
  - get_state_for_agent() : エージェントごとに状態ベクトル（自身のSoC、相手のSoC、エージェント識別子、AG充電要請、相手の充電予測値または実測値）を生成
  - step() : 同時実行モードでのエージェントの行動反映と報酬計算
  - step_sequential() : 逐次実行モードでの行動処理（先行エージェントと後攻エージェントの順次処理）
"""

import random
import numpy as np
from Config import CAPACITY, INITIAL_SOC, AG_REQUEST, EPISODE_STEPS, TOLERANCE, PENALTY_WEIGHT

class EVEnv:
    def __init__(self, capacity=CAPACITY, initial_soc=INITIAL_SOC, ag_request=AG_REQUEST, 
                episode_steps=EPISODE_STEPS, tolerance=TOLERANCE):
        self.capacity = capacity  
        self.initial_soc = initial_soc  
        # Configで設定したAG_REQUESTを使用
        self.ag_request = ag_request
        self.episode_steps = episode_steps  
        self.tolerance = tolerance
        self.reset()
    
    def reset(self):
        self.soc = {"ev1": self.initial_soc, "ev2": self.initial_soc}
        self.step_count = 0
        self.temp_actions = {}
        self.temp_predicted_first = None
        self.temp_predicted_second = None
        self.temp_first_agent = None
        self.prediction_error_sum = 0.0
        # 状態は5次元：[自身のSoC, 相手のSoC, エージェント識別子, AG充電要請, 予測または相手の充電量]
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
            return np.array([self.soc["ev1"], self.soc["ev2"], 0.0, self.ag_request, other_val], dtype=np.float32)
        elif agent == "ev2":
            return np.array([self.soc["ev2"], self.soc["ev1"], 1.0, self.ag_request, other_val], dtype=np.float32)
        else:
            raise ValueError("Unknown agent")
        
    def get_first_agent(self):
        # ステップカウントにより先行エージェントを交互に決定
        return "ev1" if ((self.step_count + 1) % 2) == 1 else "ev2"
    
    def step(self, actions):
        a1 = actions["ev1"]
        a2 = actions["ev2"]
        
        # EV1の充電更新（CAPACITYを上限とする）
        penalty_ev1 = False
        if self.soc["ev1"] + a1 > self.capacity:
            self.soc["ev1"] = self.capacity
            penalty_ev1 = True
        else:
            self.soc["ev1"] += a1
        
        # EV2の充電更新
        penalty_ev2 = False
        if self.soc["ev2"] + a2 > self.capacity:
            self.soc["ev2"] = self.capacity
            penalty_ev2 = True
        else:
            self.soc["ev2"] += a2
        
        # 現在のAG要請は状態に含まれている値を使う
        total_charge = a1 + a2
        deviation = abs(total_charge - self.ag_request)
        
        # 新しい報酬式（ピースワイズ定義）
        if deviation <= 0.5:
            reward_total = 100
        elif deviation < 1.5:
            reward_total = 20 * (1.5 - deviation)
        else:
            reward_total = 0
        
        reward_ev1 = reward_total
        reward_ev2 = reward_total
        
        # 充電更新で上限を超えた場合の固定ペナルティ（PENALTY_WEIGHTを利用）
        if penalty_ev1:
            reward_ev1 -= PENALTY_WEIGHT
        if penalty_ev2:
            reward_ev2 -= PENALTY_WEIGHT
        
        self.step_count += 1
        
        # 次ステップ用に新たなAG要請を更新（0～5の範囲からrandom.uniformで選ぶ）
        self.ag_request = random.uniform(0, 5)
        
        # 新しいAG要請が反映された状態を生成
        state_ev1 = self.get_state_for_agent("ev1", other_charge=None)
        state_ev2 = self.get_state_for_agent("ev2", other_charge=None)
        next_state = np.array([state_ev1, state_ev2])
        
        rewards = np.array([reward_ev1, reward_ev2], dtype=np.float32)
        done_flags = np.array([self.step_count >= self.episode_steps, self.step_count >= self.episode_steps], dtype=np.float32)
        info = {"total_charge": total_charge, "reward_total": reward_total, "success": (deviation < self.tolerance)}
        return next_state, rewards, done_flags, info

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
            if deviation <= 0.5:
                reward_total = 100
            elif deviation < 1.5:
                reward_total = 20 * (1.5 - deviation)
            else:
                reward_total = 0
            
            reward_ev1 = reward_total / 2.0
            reward_ev2 = reward_total / 2.0
            if penalty_ev1:
                reward_ev1 -= PENALTY_WEIGHT
            if penalty_ev2:
                reward_ev2 -= PENALTY_WEIGHT

            prediction_error = abs(self.temp_predicted_first - true_first_soc)
            self.prediction_error_sum += prediction_error

            self.step_count += 1
            
            # 次ステップ用にAG要請を更新（random.uniformを使用）
            self.ag_request = random.uniform(0, 5)
            
            state_ev1 = self.get_state_for_agent("ev1", other_charge=a2)
            state_ev2 = self.get_state_for_agent("ev2", other_charge=a1)
            next_state = np.array([state_ev1, state_ev2])
            rewards = np.array([reward_ev1, reward_ev2], dtype=np.float32)
            done_flags = np.array([self.step_count >= self.episode_steps, self.step_count >= self.episode_steps], dtype=np.float32)
            info = {
                "total_charge": total_charge,
                "reward_total": reward_total,
                "success": (deviation < self.tolerance),
                "prediction_error_sum": self.prediction_error_sum
            }
            self.temp_actions = {}
            self.temp_predicted_first = None
            self.temp_predicted_second = None
            self.temp_first_agent = None
            return next_state, rewards, done_flags, info
