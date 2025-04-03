"""
Utils.py

このファイルは、プロジェクト内で利用される補助関数や共通設定を定義します。
主な内容：
  - デバイス設定：GPUが使用可能な場合はGPU、そうでなければCPUを使用
  - プロット関数：各エピソードのデータや報酬のグラフを生成する関数
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# デバイス設定
# --------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# --------------------------
# プロット関数
# --------------------------
def plot_episode_data(steps, predicted_ev1, predicted_ev2, actual_ev1, actual_ev2, ag_requests):
    """
    48ステップ分のデータをグラフ化する関数です。
    
    Parameters:
      steps          : 各ステップのインデックス（例：np.arange(1, 49)）
      predicted_ev1  : EV1のRF予測充電量（長さ48の配列）
      predicted_ev2  : EV2のRF予測充電量（長さ48の配列）
      actual_ev1     : EV1の実際の充電量（長さ48の配列）
      actual_ev2     : EV2の実際の充電量（長さ48の配列）
      ag_requests    : AGからの要請量（長さ48の配列）
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 上段：RF予測充電量の積み上げ棒グラフ＋AG要請曲線
    axs[0].bar(steps, predicted_ev1, label="EV1 Predicted")
    axs[0].bar(steps, predicted_ev2, bottom=predicted_ev1, label="EV2 Predicted")
    axs[0].plot(steps, ag_requests, color='black', marker='o', linewidth=2, label="AG Request")
    axs[0].set_ylabel("Predicted Charge")
    axs[0].set_title("Episode Data: Predicted Charging vs AG Request")
    axs[0].legend()
    
    # 下段：実際の充電量の積み上げ棒グラフ＋AG要請曲線
    axs[1].bar(steps, actual_ev1, label="EV1 Actual")
    axs[1].bar(steps, actual_ev2, bottom=actual_ev1, label="EV2 Actual")
    axs[1].plot(steps, ag_requests, color='black', marker='o', linewidth=2, label="AG Request")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Actual Charge")
    axs[1].set_title("Episode Data: Actual Charging vs AG Request")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_episode_rewards(episode_rewards):
    """
    各エピソードの総報酬をグラフ化する関数です。
    
    Parameters:
      episode_rewards : 各エピソードの総報酬のリスト
    """
    plt.figure(figsize=(10, 5))
    episodes = np.arange(len(episode_rewards))
    plt.plot(episodes, episode_rewards, label="Episode Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode-wise Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()
