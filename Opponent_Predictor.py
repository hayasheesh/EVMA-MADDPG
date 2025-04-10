"""
Opponent_Predictor.py

このファイルは、対戦相手の充電量を予測するための機械学習モデルを用いた予測モジュールを定義します。
OpponentPredictor クラスは、入力された特徴量に基づいて対戦相手の充電量を予測します。
未学習の場合はランダムな値（0～5）を返し、学習後は学習済みモデルを用いて予測値を出力し、
その値を 0～5 の範囲に制限します。
"""

import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from Config import RF_N_ESTIMATORS

class OpponentPredictor:
    def __init__(self, model_type="gradient_boosting", name="未命名"):
        self.name = name
        self.model_type = model_type
        
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,        # 木の数を増やす
                max_depth=10,            # 適度な深さを設定
                min_samples_split=5,     # 過学習防止
                random_state=42
            )
        elif model_type == "gradient_boosting":
            # より高性能な勾配ブースティング
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        self.is_trained = False
        self.feature_means = None
        self.feature_stds = None
        self.train_count = 0  # 学習回数カウンター
        self.prediction_errors = []  # 予測誤差を保存するリスト
        
    def predict(self, X):
        if not self.is_trained:
            return random.uniform(0, 5)  # 未学習時は0～5のランダムな値を返す
        
        # 特徴量を整形
        X_arr = np.array(X).reshape(1, -1)
        
        # 特徴量の次元数チェック
        if self.feature_means is not None:
            if X_arr.shape[1] != len(self.feature_means):
                print(f"警告: {self.name}の予測で特徴量次元の不一致 - 予測時: {X_arr.shape[1]}, 学習時: {len(self.feature_means)}")
                # 警告を表示してランダム値を返す
                return random.uniform(0, 5)
                
            # 特徴量の正規化
            X_scaled = (X_arr - self.feature_means) / (self.feature_stds + 1e-8)
        else:
            X_scaled = X_arr
            
        prediction = self.model.predict(X_scaled)[0]
        # 予測値を 0～5 の範囲に制限
        return np.clip(prediction, 0, 5)
    
    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # NaNを含む行を除外
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        # 有効なサンプルが1件もない場合は学習をスキップ
        if len(X_valid) == 0:
            return

        # 特徴量の次元数が以前の学習と異なる場合はモデルを再初期化
        if self.is_trained and hasattr(self, 'feature_means') and len(self.feature_means) != X_valid.shape[1]:
            print(f"警告: {self.name}の学習で特徴量次元の変更 - 前回: {len(self.feature_means)}, 今回: {X_valid.shape[1]}")
            # モデルを再初期化
            if self.model_type == "random_forest":
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
            elif self.model_type == "gradient_boosting":
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )
            self.is_trained = False
        
        # 特徴量の統計情報を計算
        self.feature_means = X_valid.mean(axis=0)
        self.feature_stds = X_valid.std(axis=0)
        
        # 特徴量の正規化
        X_scaled = (X_valid - self.feature_means) / (self.feature_stds + 1e-8)
        
        # モデル学習
        self.model.fit(X_scaled, y_valid)
        self.is_trained = True
        self.train_count += 1
        
        # 100回ごとに学習状況をログ出力
        if self.train_count % 100 == 0:
            print(f"{self.name} - 学習回数: {self.train_count}, データ数: {len(X_valid)}, 特徴量次元: {X_valid.shape[1]}")
    
    def evaluate_prediction(self, X, y_true):
        """
        予測の正確さを評価し、誤差を記録する
        
        Parameters:
        -----------
        X : 特徴量
        y_true : 実際の値
        """
        if not self.is_trained:
            return
            
        y_pred = self.predict(X)
        error = abs(y_pred - y_true)
        self.prediction_errors.append(error)
        
        # 100個のエラーが溜まったらログ出力して平均誤差をリセット
        if len(self.prediction_errors) >= 100:
            avg_error = np.mean(self.prediction_errors)
            print(f"{self.name} - 直近100件の平均予測誤差: {avg_error:.4f}")
            self.prediction_errors = []  # リセット


# 対戦相手の充電量予測器のインスタンスを生成
rf_predictor_ev1 = OpponentPredictor(model_type="gradient_boosting", name="EV1予測器")
rf_predictor_ev2 = OpponentPredictor(model_type="gradient_boosting", name="EV2予測器")
