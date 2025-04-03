"""
Opponent_Predictor.py

このファイルは、対戦相手の充電量を予測するためのランダムフォレストを用いた予測モジュールを定義します。
OpponentPredictorRF クラスは、入力された特徴量に基づいて対戦相手の充電量を予測します。
未学習の場合はランダムな値（0～5）を返し、学習後は学習済みモデルを用いて予測値を出力し、
その値を 0～5 の範囲に制限します。
"""

import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Config import RF_N_ESTIMATORS

class OpponentPredictorRF:
    def __init__(self, n_estimators=RF_N_ESTIMATORS):
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.is_trained = False

    def predict(self, X):
        if not self.is_trained:
            return random.uniform(0, 5)  # 未学習時は0～5のランダムな値を返す
        prediction = self.model.predict([X])[0]
        # 予測値を 0～5 の範囲に制限（softplus等により0以上になる前提）
        return np.clip(prediction, 0, 5)

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

# 対戦相手の充電量予測器のインスタンスを生成
rf_predictor_ev1 = OpponentPredictorRF(n_estimators=RF_N_ESTIMATORS)
rf_predictor_ev2 = OpponentPredictorRF(n_estimators=RF_N_ESTIMATORS)
