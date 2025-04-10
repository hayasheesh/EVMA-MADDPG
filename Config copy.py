"""
Config.py

このファイルは、プロジェクト全体で共通して使用されるハイパーパラメータや設定を定義します。
以下のパラメータを含みます：
  - エピソード数、学習率、ネットワーク構造、Replay Buffer のサイズなどの学習関連パラメータ
  - RandomForest や EV 環境、ノイズ生成に関するパラメータ

"""

import random

# --------------------------
# ハイパーパラメータ定義
# --------------------------

# 共通
NUM_EPISODES = 20000      # 総エピソード数

# RandomForest 用ハイパーパラメータ
RF_TRAIN_INTERVAL = 100     # RF学習呼び出し間隔（エピソード毎）
RF_N_ESTIMATORS = 10         # RandomForestの決定木の数

# MADDPG 用ハイパーパラメータ
LR_ACTOR = 1e-3              # Actor の学習率
LR_CRITIC = 1e-3             # Critic の学習率
STATE_DIM = 4                # 環境状態の次元
ACTION_DIM = 1               # 行動の次元
NUM_AGENTS = 2               # エージェント数
MADDPG_HIDDEN_SIZE = 64      # ネットワークの隠れ層のサイズ
GAMMA = 0.95                 # 割引率
TAU = 0.01                   # ソフトアップデート係数
BATCH_SIZE = 128             # バッチサイズ
MEMORY_SIZE = int(1e6)       # Replay Buffer のサイズ

# EV環境用ハイパーパラメータ
CAPACITY = 100.0             # EVの容量
INITIAL_SOC = 0.0            # 初期 SoC 値
AG_REQUEST = random.uniform(0, 5)  # AGからの充電要請（初期値をランダムに設定）
EPISODE_STEPS = 48           # 1エピソード内のステップ数
TOLERANCE = 0.1              # 成功判定用の許容誤差
PENALTY_WEIGHT = 100.0       # 充電上限超過時のペナルティ

# Ornstein-Uhlenbeck Noise 用ハイパーパラメータ
OU_MU = 0.0                  # 平均
OU_THETA = 0.15              # 速度（戻りの速さ）
OU_SIGMA = 0.2               # ノイズの振幅
OU_DT = 1e-2                 # 時間刻み

# Critic 初期化用ハイパーパラメータ
CRITIC_INIT_W = 3e-3         # Critic 最終層の重み初期化幅

# 報酬計算の閾値設定
TOLERANCE_NARROW = 0.5  # 厳密な許容範囲（この範囲内なら最大報酬）
TOLERANCE_WIDE = 1.5    # 緩い許容範囲（この範囲を超えると報酬なし）
