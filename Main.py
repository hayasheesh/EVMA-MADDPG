import numpy as np
from Config import NUM_EPISODES, STATE_DIM, ACTION_DIM, NUM_AGENTS, EPISODE_STEPS, INITIAL_SOC, CAPACITY, AG_REQUEST, RF_TRAIN_INTERVAL
from EVEnv import EVEnv
from Agent import MADDPG
from Opponent_Predictor import rf_predictor_ev1, rf_predictor_ev2
from Utils import plot_episode_data, plot_episode_rewards



# 学習モード。ノイズを加えた行動を逐次実行モードで実行
def train_maddpg_sequential(num_episodes=NUM_EPISODES):
    # EV環境およびエージェントの初期化
    env = EVEnv(capacity=CAPACITY, initial_soc=INITIAL_SOC, ag_request=AG_REQUEST, 
                episode_steps=EPISODE_STEPS)
    agent = MADDPG(STATE_DIM, ACTION_DIM, num_agents=NUM_AGENTS)
    
    # RFの学習用データリスト（各エージェントごと）
    rf_data_ev1 = []  # EV1用（EV2の戦略を学習）
    rf_labels_ev1 = []
    rf_data_ev2 = []  # EV2用（EV1の戦略を学習）
    rf_labels_ev2 = []
    
    # 履歴情報の初期化
    history_length = 5
    ev1_action_history = [0] * history_length  # 初期値は0で埋める
    ev2_action_history = [0] * history_length
    ag_request_history = [0] * history_length
    
    # データ管理設定
    max_data_size = 20000  # 最大データサイズ
    
    # 特徴量の次元数をログ出力
    feature_dim = 5 + (history_length * 2)  # 基本特徴量5つ + 行動履歴 + AG要請履歴
    
    episode_rewards = []
    save_folder = r"C:\Users\hayas\Desktop\EVMA_Local\結果保存"  # グラフ画像の保存先フォルダ
    
    for ep in range(num_episodes):
        # 各エピソード内のデータ記録リスト（プロット用）
        predicted_ev1_steps = []
        predicted_ev2_steps = []
        actual_ev1_steps = []
        actual_ev2_steps = []
        ag_requests_steps = []
        
        # 予測値の初期化
        predicted_ev1_prev = 0
        predicted_ev2_prev = 0
        
        # エピソードごとに履歴をリセット
        ev1_action_history = [0] * history_length
        ev2_action_history = [0] * history_length
        ag_request_history = [0] * history_length
        
        state = env.reset()
        
        for t in range(env.episode_steps):
            current_request = env.ag_request
            
            # --- ステップ1：先行エージェントのRF予測（相手の充電量予測） ---
            first_agent = env.get_first_agent()
            if first_agent == "ev1":
                # EV1が先行の場合、EV2の行動を予測
                # 拡張特徴量
                features_first = [
                    state[0][1],           # 相手（EV2）のSoC
                    state[0][2],           # AG要請値
                    state[0][0],           # 自分（EV1）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[0][0] + state[0][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_first.extend(ev2_action_history)  # 相手の行動履歴
                features_first.extend(ag_request_history)  # AG要請の履歴
                
                pred_first = rf_predictor_ev1.predict(features_first)
                env.temp_predicted_first = pred_first
                env.temp_first_agent = "ev1"
                state_first = env.get_state_for_agent("ev1", other_charge=pred_first)
                action_first = agent.get_actions(np.array([state_first, state[1]]), add_noise=True)[0][0]
            else:
                # EV2が先行の場合、EV1の行動を予測
                # 拡張特徴量
                features_first = [
                    state[1][1],           # 相手（EV1）のSoC
                    state[1][2],           # AG要請値
                    state[1][0],           # 自分（EV2）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[1][0] + state[1][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_first.extend(ev1_action_history)  # 相手の行動履歴
                features_first.extend(ag_request_history)  # AG要請の履歴
                
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
                # EV1が後攻の場合、EV2の行動を予測
                # 拡張特徴量
                features_second = [
                    state[0][1],           # 相手（EV2）のSoC
                    state[0][2],           # AG要請値
                    state[0][0],           # 自分（EV1）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[0][0] + state[0][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_second.extend(ev2_action_history)  # 相手の行動履歴
                features_second.extend(ag_request_history)  # AG要請の履歴
                
                pred_second = rf_predictor_ev1.predict(features_second)
            else:
                # EV2が後攻の場合、EV1の行動を予測
                # 拡張特徴量
                features_second = [
                    state[1][1],           # 相手（EV1）のSoC
                    state[1][2],           # AG要請値
                    state[1][0],           # 自分（EV2）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[1][0] + state[1][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_second.extend(ev1_action_history)  # 相手の行動履歴
                features_second.extend(ag_request_history)  # AG要請の履歴
                
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
            
            # RF学習用データの蓄積
            # EV1の予測器は相手(EV2)の行動を予測するため、EV1の状態とEV2の行動を記録
            # 拡張特徴量
            features_ev1 = [
                state[0][1],           # 相手（EV2）のSoC
                state[0][2],           # AG要請値
                state[0][0],           # 自分（EV1）のSoC
                env.step_count,        # 現在のステップ数
                # 環境要因
                current_request / max(1.0, (state[0][0] + state[0][1])),  # 全体SoCに対する要請の割合
            ]
            # 履歴情報を追加（EV2の行動履歴とAG要請の履歴）
            # 注意: 予測時と同じ履歴を使用する
            features_ev1.extend(ev2_action_history)  # 相手の行動履歴（最新のものも含む）
            features_ev1.extend(ag_request_history)  # AG要請の履歴（最新のものも含む）
            
            rf_data_ev1.append(features_ev1)
            rf_labels_ev1.append(action_ev2)  # EV2の実際の行動を記録
            
            # EV2の予測器は相手(EV1)の行動を予測するため、EV2の状態とEV1の行動を記録
            # 拡張特徴量
            features_ev2 = [
                state[1][1],           # 相手（EV1）のSoC
                state[1][2],           # AG要請値
                state[1][0],           # 自分（EV2）のSoC
                env.step_count,        # 現在のステップ数
                # 環境要因
                current_request / max(1.0, (state[1][0] + state[1][1])),  # 全体SoCに対する要請の割合
            ]
            # 履歴情報を追加（EV1の行動履歴とAG要請の履歴）
            # 注意: 予測時と同じ履歴を使用する
            features_ev2.extend(ev1_action_history)  # 相手の行動履歴（最新のものも含む）
            features_ev2.extend(ag_request_history)  # AG要請の履歴（最新のものも含む）
            
            rf_data_ev2.append(features_ev2)
            rf_labels_ev2.append(action_ev1)  # EV1の実際の行動を記録
            
            # 予測と実際の行動を記録（プロット用）
            predicted_ev1_steps.append(predicted_ev1_this_step)
            predicted_ev2_steps.append(predicted_ev2_this_step)
            actual_ev1_steps.append(action_ev1)
            actual_ev2_steps.append(action_ev2)
            ag_requests_steps.append(current_request)
            
            # 後攻エージェントの行動実行（充電更新と報酬計算）
            next_state, rewards, done, info = env.step_sequential(second_agent, action_second)
            
            # エージェントの学習用にデータをReplayBufferに保存
            combined_actions = np.array([[action_ev1], [action_ev2]])
            agent.buffer.cache(state, next_state, combined_actions, rewards, done)
            
            # 行動履歴を更新（次のステップの予測のために現在のステップの行動を記録）
            ev1_action_history.append(action_ev1)
            ev2_action_history.append(action_ev2)
            ag_request_history.append(current_request)
            
            # 履歴の長さを制限
            ev1_action_history = ev1_action_history[-history_length:]
            ev2_action_history = ev2_action_history[-history_length:]
            ag_request_history = ag_request_history[-history_length:]
            
            state = next_state
            if np.all(done):
                break

            # 12ステップごとにupdateを呼び出す
            if (t + 1) % 1 == 0:
                agent.update()
        
        episode_rewards.append(np.sum(rewards))
        
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}: Average Reward: {avg_reward:.3f}")
        
        # データ量の管理とRF学習の実行
        if (ep + 1) % RF_TRAIN_INTERVAL == 0:
            # データが最大サイズを超えたら、古いデータを削除
            if len(rf_data_ev1) > max_data_size:
                excess = len(rf_data_ev1) - max_data_size
                rf_data_ev1 = rf_data_ev1[excess:]
                rf_labels_ev1 = rf_labels_ev1[excess:]
            
            if len(rf_data_ev2) > max_data_size:
                excess = len(rf_data_ev2) - max_data_size
                rf_data_ev2 = rf_data_ev2[excess:]
                rf_labels_ev2 = rf_labels_ev2[excess:]
            
            # 十分なデータがある場合のみ学習を実行
            if len(rf_data_ev1) > 10:
                rf_predictor_ev1.train(np.array(rf_data_ev1), np.array(rf_labels_ev1))
            
            if len(rf_data_ev2) > 10:
                rf_predictor_ev2.train(np.array(rf_data_ev2), np.array(rf_labels_ev2))
    
        
        # 1000エピソードごとにテストモードを実行
        if (ep + 1) % 1000 == 0:
            print(f"Running test mode at episode {ep+1}...")
            
            # プロット用にこのエピソードのデータを整形
            steps = np.arange(1, env.episode_steps + 1)
            min_len = min(len(predicted_ev1_steps), len(actual_ev1_steps), len(ag_requests_steps))
            
            # 予測値の配列をそのまま使用（修正なし）
            predicted_ev1_steps_plot = predicted_ev1_steps[:min_len]
            predicted_ev2_steps_plot = predicted_ev2_steps[:min_len]
            actual_ev1_steps_plot = actual_ev1_steps[:min_len]
            actual_ev2_steps_plot = actual_ev2_steps[:min_len]
            ag_requests_steps_plot = ag_requests_steps[:min_len]
            steps_plot = steps[:min_len]
            
            # 現在のエピソードのデータをプロット
            init_soc_ev1 = env.initial_soc["ev1"]
            init_soc_ev2 = env.initial_soc["ev2"]
            plot_episode_data(steps_plot, predicted_ev1_steps_plot, predicted_ev2_steps_plot, actual_ev1_steps_plot, actual_ev2_steps_plot, ag_requests_steps_plot, init_soc_ev1, init_soc_ev2, f"train_{ep+1}", save_folder)
            
            # テストモードを実行
            test_episode_rewards = test_maddpg_sequential(agent, env, rf_predictor_ev1, rf_predictor_ev2, save_folder, ep + 1)
            print(f"Test mode average reward: {np.mean(test_episode_rewards):.3f}")
    
    return agent, env, episode_rewards


# テストモード。ノイズを加えない行動を逐次実行モードで実行
def test_maddpg_sequential(agent, env, rf_predictor_ev1, rf_predictor_ev2, save_folder, current_episode, num_test_episodes=10):
    """
    テストモードを実行する関数
    
    Parameters:
    -----------
    agent : MADDPG
        学習済みのMADDPGエージェント
    env : EVEnv
        環境インスタンス
    rf_predictor_ev1 : RandomForestPredictor
        EV1用の学習済みRF予測器
    rf_predictor_ev2 : RandomForestPredictor
        EV2用の学習済みRF予測器
    save_folder : str
        結果を保存するフォルダパス
    current_episode : int
        現在の学習エピソード番号（ファイル名に使用）
    num_test_episodes : int
        テストで実行するエピソード数
    
    Returns:
    --------
    test_episode_rewards : list
        テストエピソードの報酬リスト
    """
    test_episode_rewards = []
    
    # 履歴情報の初期化
    history_length = 5
    
    for test_ep in range(num_test_episodes):
        # 各エピソード内のデータ記録リスト（プロット用）
        predicted_ev1_steps = []
        predicted_ev2_steps = []
        actual_ev1_steps = []
        actual_ev2_steps = []
        ag_requests_steps = []
        
        # 予測値の初期化
        predicted_ev1_prev = 0
        predicted_ev2_prev = 0
        
        # エピソードごとに履歴をリセット
        ev1_action_history = [0] * history_length
        ev2_action_history = [0] * history_length
        ag_request_history = [0] * history_length
        
        state = env.reset()
        
        for t in range(env.episode_steps):
            current_request = env.ag_request
            
            # --- ステップ1：先行エージェントのRF予測（相手の充電量予測） ---
            first_agent = env.get_first_agent()
            if first_agent == "ev1":
                # EV1が先行の場合、EV2の行動を予測
                # 拡張特徴量
                features_first = [
                    state[0][1],           # 相手（EV2）のSoC
                    state[0][2],           # AG要請値
                    state[0][0],           # 自分（EV1）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[0][0] + state[0][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_first.extend(ev2_action_history)  # 相手の行動履歴
                features_first.extend(ag_request_history)  # AG要請の履歴
                
                pred_first = rf_predictor_ev1.predict(features_first)
                env.temp_predicted_first = pred_first
                env.temp_first_agent = "ev1"
                state_first = env.get_state_for_agent("ev1", other_charge=pred_first)
                action_first = agent.get_actions(np.array([state_first, state[1]]), add_noise=False)[0][0]
            else:
                # EV2が先行の場合、EV1の行動を予測
                # 拡張特徴量
                features_first = [
                    state[1][1],           # 相手（EV1）のSoC
                    state[1][2],           # AG要請値
                    state[1][0],           # 自分（EV2）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[1][0] + state[1][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_first.extend(ev1_action_history)  # 相手の行動履歴
                features_first.extend(ag_request_history)  # AG要請の履歴
                
                pred_first = rf_predictor_ev2.predict(features_first)
                env.temp_predicted_first = pred_first
                env.temp_first_agent = "ev2"
                state_first = env.get_state_for_agent("ev2", other_charge=pred_first)
                action_first = agent.get_actions(np.array([state[0], state_first]), add_noise=False)[1][0]
            
            # 先行エージェントの行動を逐次実行モードで反映
            _ = env.step_sequential(env.temp_first_agent, action_first)
            first_agent_used = env.temp_first_agent
            
            # --- ステップ2：後攻エージェントのRF予測（相手の充電量予測） ---
            second_agent = "ev2" if first_agent_used == "ev1" else "ev1"
            if second_agent == "ev1":
                # EV1が後攻の場合、EV2の行動を予測
                # 拡張特徴量
                features_second = [
                    state[0][1],           # 相手（EV2）のSoC
                    state[0][2],           # AG要請値
                    state[0][0],           # 自分（EV1）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[0][0] + state[0][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_second.extend(ev2_action_history)  # 相手の行動履歴
                features_second.extend(ag_request_history)  # AG要請の履歴
                
                pred_second = rf_predictor_ev1.predict(features_second)
            else:
                # EV2が後攻の場合、EV1の行動を予測
                # 拡張特徴量
                features_second = [
                    state[1][1],           # 相手（EV1）のSoC
                    state[1][2],           # AG要請値
                    state[1][0],           # 自分（EV2）のSoC
                    env.step_count,        # 現在のステップ数
                    # 環境要因
                    current_request / max(1.0, (state[1][0] + state[1][1])),  # 全体SoCに対する要請の割合
                ]
                # 履歴情報を追加
                features_second.extend(ev1_action_history)  # 相手の行動履歴
                features_second.extend(ag_request_history)  # AG要請の履歴
                
                pred_second = rf_predictor_ev2.predict(features_second)
            env.temp_predicted_second = pred_second
            
            # --- ステップ3：後攻エージェントは実測値を利用して行動決定 ---
            state_second = env.get_state_for_agent(second_agent, other_charge=env.temp_actions.get(first_agent_used, -1.0))
            if second_agent == "ev1":
                action_second = agent.get_actions(np.array([state_second, state[1]]), add_noise=False)[0][0]
            else:
                action_second = agent.get_actions(np.array([state[0], state_second]), add_noise=False)[1][0]
            
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
            
            # 予測と実際の行動を記録（プロット用）
            predicted_ev1_steps.append(predicted_ev1_this_step)
            predicted_ev2_steps.append(predicted_ev2_this_step)
            actual_ev1_steps.append(action_ev1)
            actual_ev2_steps.append(action_ev2)
            ag_requests_steps.append(current_request)
            
            # 後攻エージェントの行動実行（充電更新と報酬計算）
            next_state, rewards, done, info = env.step_sequential(second_agent, action_second)
            
            # エージェントの学習用にデータをReplayBufferに保存
            combined_actions = np.array([[action_ev1], [action_ev2]])
            agent.buffer.cache(state, next_state, combined_actions, rewards, done)
            
            # 行動履歴を更新（次のステップの予測のために現在のステップの行動を記録）
            ev1_action_history.append(action_ev1)
            ev2_action_history.append(action_ev2)
            ag_request_history.append(current_request)
            
            # 履歴の長さを制限
            ev1_action_history = ev1_action_history[-history_length:]
            ev2_action_history = ev2_action_history[-history_length:]
            ag_request_history = ag_request_history[-history_length:]
            
            state = next_state
            if np.all(done):
                break
        
        # エピソード終了時に最後の予測値を追加（削除）
        
        test_episode_rewards.append(np.sum(rewards))
        
        # テストエピソードのデータを保存
        steps = np.arange(1, env.episode_steps + 1)
        init_soc_ev1 = env.initial_soc["ev1"]
        init_soc_ev2 = env.initial_soc["ev2"]
        
        # データ長が一致するように調整
        min_len = min(len(predicted_ev1_steps), len(actual_ev1_steps), len(ag_requests_steps))
        
        # 予測値の配列をそのまま使用（修正なし）
        predicted_ev1_steps = predicted_ev1_steps[:min_len]
        predicted_ev2_steps = predicted_ev2_steps[:min_len]
        actual_ev1_steps = actual_ev1_steps[:min_len]
        actual_ev2_steps = actual_ev2_steps[:min_len]
        ag_requests_steps = ag_requests_steps[:min_len]
        steps = steps[:min_len]
        
        # 予測と実際の行動の差異を確認 (100エピソード目のみ)
        if current_episode >= 1000 and test_ep == 0:
            avg_diff_ev1 = np.mean(np.abs(np.array(predicted_ev1_steps) - np.array(actual_ev1_steps)))
            avg_diff_ev2 = np.mean(np.abs(np.array(predicted_ev2_steps) - np.array(actual_ev2_steps)))
            print(f"予測精度 - EV1の平均誤差: {avg_diff_ev1:.4f}, EV2の平均誤差: {avg_diff_ev2:.4f}")
        
        plot_episode_data(steps, predicted_ev1_steps, predicted_ev2_steps, actual_ev1_steps, actual_ev2_steps, ag_requests_steps, init_soc_ev1, init_soc_ev2, f"test_{current_episode}_{test_ep+1}", save_folder)
    
    # テストエピソードの報酬を保存
    plot_episode_rewards(test_episode_rewards, save_folder, prefix=f"test_{current_episode}")
    
    return test_episode_rewards


if __name__ == '__main__':
    agent, env, episode_rewards = train_maddpg_sequential(num_episodes=NUM_EPISODES)
