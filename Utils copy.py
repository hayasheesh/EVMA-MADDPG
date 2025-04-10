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
def plot_episode_data(steps, predicted_ev1, predicted_ev2, actual_ev1, actual_ev2, ag_requests, init_soc_ev1, init_soc_ev2, episode_number, save_folder):
    """
    各エピソードのデータ（予測充電量/実際の充電量、AG要請）をプロットし、指定フォルダに保存する関数です。
    タイトルは「episode_{episode_number}(EV1__{init_soc_ev1:.1f},_EV2__{init_soc_ev2:.1f})」の形式になります。
    
    Parameters:
      steps         : 各ステップの番号のリスト
      predicted_ev1 : EV1の予測充電量リスト
      predicted_ev2 : EV2の予測充電量リスト
      actual_ev1    : EV1の実際の充電量リスト
      actual_ev2    : EV2の実際の充電量リスト
      ag_requests   : 各ステップでのAG要請リスト
      init_soc_ev1  : EV1の初期SoC
      init_soc_ev2  : EV2の初期SoC
      episode_number: 出力時の最後のエピソード番号
      save_folder   : グラフ画像の保存先フォルダ
    """
    # フォルダが存在しない場合は作成
    os.makedirs(save_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # タイトル用の共通部分：呼び出し側から渡されたエピソード番号を使用
    title_base = f"episode_{episode_number}(EV1__{init_soc_ev1:.1f},_EV2__{init_soc_ev2:.1f})"
    
    # 予測値の配列をそのまま使用（修正なし）
    predicted_ev1_plot = predicted_ev1
    predicted_ev2_plot = predicted_ev2
    
    # 上段: RF予測充電量の積み上げ棒グラフ＋AG要請曲線
    # 予測値は現在のステップの予測を表示するため、同じsteps配列を使用
    axs[0].bar(steps, predicted_ev1_plot, label="EV1 Predicted")
    axs[0].bar(steps, predicted_ev2_plot, bottom=predicted_ev1_plot, label="EV2 Predicted")
    axs[0].plot(steps, ag_requests, color='black', marker='o', linewidth=2, label="AG Request")
    axs[0].set_ylabel("Predicted Charge")
    axs[0].set_title(title_base + " - Predicted Charging vs AG Request")
    axs[0].legend()
    
    # 下段: 実際の充電量の積み上げ棒グラフ＋AG要請曲線
    axs[1].bar(steps, actual_ev1, label="EV1 Actual")
    axs[1].bar(steps, actual_ev2, bottom=actual_ev1, label="EV2 Actual")
    axs[1].plot(steps, ag_requests, color='black', marker='o', linewidth=2, label="AG Request")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Actual Charge")
    axs[1].set_title(title_base + " - Actual Charging vs AG Request")
    axs[1].legend()
    
    plt.tight_layout()
    
    # ファイル名は上段のタイトルを基に生成
    filename = safe_filename(title_base) + ".png"
    file_path = os.path.join(save_folder, filename)
    plt.savefig(file_path)
    plt.close(fig)
    print(f"グラフを保存しました: {file_path}")


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
