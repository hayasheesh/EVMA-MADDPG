"""
Main.py

このファイルは、プロジェクトのエントリーポイントです。
以下のモジュールを読み込み、MADDPGエージェントによる充電割当シミュレーションおよび学習ループを実行します。

使用モジュール：
  - Config      : ハイパーパラメータの設定
  - EVEnv       : EV環境クラス（充電シミュレーションのロジック）
  - Opponent_Predictor : 対戦相手の充電量予測モジュール（RandomForestを使用）
  - Agent       : MADDPGエージェント（Actor, Critic, 学習ロジック）
  - Utils       : 共通設定（デバイス）およびプロット関数

主な処理：
  - 環境およびエージェントの初期化
  - 各エピソードにおける逐次実行モードでの行動取得、報酬計算、学習更新
  - 一定エピソード毎にRF予測モジュールの学習とグラフ描画
"""

import numpy as np
from Config import NUM_EPISODES, STATE_DIM, ACTION_DIM, NUM_AGENTS, EPISODE_STEPS, INITIAL_SOC, CAPACITY, AG_REQUEST, RF_TRAIN_INTERVAL
from EVEnv import EVEnv
from Agent import MADDPG
from Opponent_Predictor import rf_predictor_ev1, rf_predictor_ev2
from Utils import plot_episode_data, plot_episode_rewards

def train_maddpg_sequential(num_episodes=NUM_EPISODES):
    # EV環境およびエージェントの初期化
    env = EVEnv(capacity=CAPACITY, initial_soc=INITIAL_SOC, ag_request=AG_REQUEST, 
                episode_steps=EPISODE_STEPS, tolerance=0.1)
    agent = MADDPG(STATE_DIM, ACTION_DIM, num_agents=NUM_AGENTS)
    
    # RFの学習用データリスト（各エージェントごと）
    rf_data_ev1 = []  # EV1用（EV2の戦略を学習）
    rf_labels_ev1 = []
    rf_data_ev2 = []  # EV2用（EV1の戦略を学習）
    rf_labels_ev2 = []
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        # RF学習用データの定期リセット（RF_TRAIN_INTERVAL毎）
        if ep > 0 and (ep % RF_TRAIN_INTERVAL == 0):
            rf_data_ev1.clear()
            rf_labels_ev1.clear()
            rf_data_ev2.clear()
            rf_labels_ev2.clear()
        
        # 各エピソード内のデータ記録リスト（プロット用）
        predicted_ev1_steps = []
        predicted_ev2_steps = []
        actual_ev1_steps = []
        actual_ev2_steps = []
        ag_requests_steps = []
        
        state = env.reset()
        
        for t in range(env.episode_steps):
            current_request = env.ag_request
            
            # --- ステップ1：先行エージェントのRF予測（相手の充電量予測） ---
            first_agent = env.get_first_agent()
            if first_agent == "ev1":
                features_first = [state[0][1], state[0][3]]
                pred_first = rf_predictor_ev1.predict(features_first)
                env.temp_predicted_first = pred_first
                env.temp_first_agent = "ev1"
                state_first = env.get_state_for_agent("ev1", other_charge=pred_first)
                action_first = agent.get_actions(np.array([state_first, state[1]]), add_noise=True)[0][0]
            else:
                features_first = [state[1][1], state[1][3]]
                pred_first = rf_predictor_ev2.predict(features_first)
                env.temp_predicted_first = pred_first
                env.temp_first_agent = "ev2"
                state_first = env.get_state_for_agent("ev2", other_charge=pred_first)
                action_first = agent.get_actions(np.array([state[0], state_first]), add_noise=True)[1][0]
            
            # 先行エージェントの行動を逐次実行モードで反映
            _ = env.step_sequential(env.temp_first_agent, action_first)
            first_agent_used = env.temp_first_agent
            
            # --- ステップ2：後攻エージェントのRF予測（相手の充電量予測） ---
            second_agent = "ev2" if first_agent_used == "ev1" else "ev1"
            if second_agent == "ev1":
                features_second = [state[0][1], state[0][3]]
                pred_second = rf_predictor_ev1.predict(features_second)
            else:
                features_second = [state[1][1], state[1][3]]
                pred_second = rf_predictor_ev2.predict(features_second)
            env.temp_predicted_second = pred_second
            
            # --- ステップ3：後攻エージェントは実測値を利用して行動決定 ---
            state_second = env.get_state_for_agent(second_agent, other_charge=env.temp_actions.get(first_agent_used, -1.0))
            if second_agent == "ev1":
                action_second = agent.get_actions(np.array([state_second, state[1]]), add_noise=True)[0][0]
            else:
                action_second = agent.get_actions(np.array([state[0], state_second]), add_noise=True)[1][0]
            
            # --- 各エージェントの行動およびRF予測値の整理 ---
            if first_agent_used == "ev1":
                action_ev1 = action_first
                action_ev2 = action_second
                predicted_ev1_this_step = pred_first
                predicted_ev2_this_step = pred_second
            else:
                action_ev2 = action_first
                action_ev1 = action_second
                predicted_ev2_this_step = pred_first
                predicted_ev1_this_step = pred_second
            
            # 各ステップのデータ記録（プロット用）
            predicted_ev1_steps.append(predicted_ev1_this_step)
            predicted_ev2_steps.append(predicted_ev2_this_step)
            actual_ev1_steps.append(action_ev1)
            actual_ev2_steps.append(action_ev2)
            ag_requests_steps.append(current_request)
            
            # RF学習用データの蓄積
            rf_data_ev1.append([state[0][1], state[0][3]])
            rf_labels_ev1.append(action_ev2)
            rf_data_ev2.append([state[1][1], state[1][3]])
            rf_labels_ev2.append(action_ev1)
            
            # 後攻エージェントの行動実行（充電更新と報酬計算）
            next_state, rewards, done, info = env.step_sequential(second_agent, action_second)
            
            combined_actions = np.array([[action_ev1], [action_ev2]])
            agent.buffer.cache(state, next_state, combined_actions, rewards, done)
            agent.update()
            
            state = next_state
            if np.all(done):
                break
        
        episode_rewards.append(np.sum(rewards))
        
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}: Average Reward: {avg_reward:.3f}")
        
        if (ep + 1) % RF_TRAIN_INTERVAL == 0:
            if len(rf_data_ev1) > 0:
                rf_predictor_ev1.train(np.array(rf_data_ev1), np.array(rf_labels_ev1))
            if len(rf_data_ev2) > 0:
                rf_predictor_ev2.train(np.array(rf_data_ev2), np.array(rf_labels_ev2))
    
        if (ep + 1) % 1000 == 0:
            steps = np.arange(1, env.episode_steps + 1)
            plot_episode_data(steps, predicted_ev1_steps, predicted_ev2_steps, actual_ev1_steps, actual_ev2_steps, ag_requests_steps)
    
    plot_episode_rewards(episode_rewards)
    
    return agent, env, episode_rewards

if __name__ == '__main__':
    agent, env, episode_rewards = train_maddpg_sequential(num_episodes=NUM_EPISODES)
