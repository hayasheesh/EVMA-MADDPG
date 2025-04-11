"""
Utils.py

このファイルは、プロジェクト内で利用される補助関数や共通設定を定義します。
主な内容：
  - デバイス設定：GPUが使用可能な場合はGPU、そうでなければCPUを使用
  - プロット関数：各エピソードのデータや報酬のグラフを生成し、指定フォルダに保存する関数
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# デバイス設定
# --------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# --------------------------
# ファイル名用のサニタイズ関数
# --------------------------
def safe_filename(title):
    # ファイル名として不適切な文字（例：スペース、コロン、スラッシュなど）をアンダースコアに置換
    invalid_chars = [':', '/', '\\', '*', '?', '"', '<', '>', '|']
    for ch in invalid_chars:
        title = title.replace(ch, "_")
    # 余分なスペースはアンダースコアに変換
    return title.strip().replace(" ", "_")

# --------------------------
# プロット関数
# --------------------------
def plot_episode_data(data, save_path, episode_num=None, initial_soc=None):
    """
    MADDPGのみのモード用のプロット関数
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = range(len(data['ag_requests']))
    
    # 実際の充電量の積み上げ棒グラフ＋AG要請曲線
    # 充電量を配列に変換して確実にnumpy配列として処理
    ev1_data = np.array(data['actual_ev1'])
    ev2_data = np.array(data['actual_ev2'])
    ev3_data = np.array(data['actual_ev3'])
    
    ax.bar(steps, ev1_data, label="EV1 Actual", color='tab:blue')
    ax.bar(steps, ev2_data, bottom=ev1_data, label="EV2 Actual", color='tab:orange')
    
    # EV1とEV2の合計を計算して、その上にEV3をプロット
    ev12_sum = ev1_data + ev2_data
    ax.bar(steps, ev3_data, bottom=ev12_sum, label="EV3 Actual", color='tab:green')
    
    ax.plot(steps, data['ag_requests'], color='black', marker='o', linewidth=2, label="AG Request")
    ax.set_xlabel("Step")
    ax.set_ylabel("Charge")
    
    # タイトルの設定
    if episode_num is not None and initial_soc is not None:
        title = f"MADDPG Only - Episode:{episode_num}, EV1:{initial_soc['ev1']:.1f}%, EV2:{initial_soc['ev2']:.1f}%, EV3:{initial_soc['ev3']:.1f}%"
    else:
        title = "MADDPG Only - Actual Charging vs AG Request"
    
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"グラフを保存しました: {save_path}")

def plot_episode_rewards(episode_rewards, save_folder, prefix=""):
    """
    エピソード報酬のグラフを生成し、指定フォルダに保存する関数です。
    
    Parameters:
      episode_rewards : 各エピソードの報酬リスト
      save_folder    : グラフ画像の保存先フォルダ
      prefix         : ファイル名のプレフィックス（テストモード用）
    """
    aggregated_rewards = []
    aggregated_episodes = []
    for i in range(0, len(episode_rewards), 100):
        block = episode_rewards[i:i+100]
        total = sum(block)
        aggregated_rewards.append(total)
        # 各ブロックの最後のエピソード番号を横軸に（例：ブロックが50エピソードの場合は50となる）
        aggregated_episodes.append(i + len(block))
    
    plt.figure(figsize=(10, 5))
    plt.plot(aggregated_episodes, aggregated_rewards, marker='o', linestyle='-', label="Total Reward per 100 Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (100 episodes)")
    title = "Aggregated Total Reward every 100 Episodes"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    filename = safe_filename(title) + ".png"
    if prefix:
        filename = f"{prefix}_{filename}"
    file_path = os.path.join(save_folder, filename)
    plt.savefig(file_path)
    plt.close()
    print(f"グラフを保存しました: {file_path}")
