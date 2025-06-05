from copy import deepcopy

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OneMaxEnv(gym.Env):
    """
    One Max問題: n次元のバイナリベクトルで1の数を最大化する
    行動: 各ビット位置をフリップ (0→1 または 1→0)
    """

    def __init__(self, n_bits: int, initial_ones_ratio: float, n_max_steps: int):
        """
        One Max環境の初期化

        Args:
            n_bits (int): ビット数
            initial_ones_ratio (float): 初期状態での1の比率
            n_max_steps (int): 最大ステップ数
        """
        super().__init__()
        self.n_bits = n_bits
        self.initial_ones_ratio = initial_ones_ratio
        self.n_max_steps = n_max_steps

        self.step_count = 0
        self.bits = np.zeros(n_bits, dtype=np.int32)
        self.initial_bits = deepcopy(self.bits)

        # 行動空間: n_bitsのビット位置
        self.action_space = spaces.Discrete(n_bits)
        # 観測空間: バイナリベクトル
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_bits,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        環境のリセット

        Args:
            seed (int | None): シード値
            options (dict | None): オプション

        Returns:
            tuple[np.ndarray, dict]: 観測, 情報
        """
        super().reset(seed=seed)

        # ステップ数をリセット
        self.step_count = 0

        # 初期状態をランダムに生成
        self.bits = np.random.choice([0, 1], size=self.n_bits, p=[1 - self.initial_ones_ratio, self.initial_ones_ratio])
        self.initial_bits = deepcopy(self.bits)

        obs = self.bits.astype(np.float32)
        return obs, {}

    def step(self, action: int):
        """
        1ステップの実行

        Args:
            action (int): 実行するアクション

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: 観測, 報酬, 終了, 切り捨て, 情報
        """
        reward = 0
        terminated = False
        truncated = False
        self.step_count += 1

        # アクションを実行
        if self.bits[action] == 0:
            self.bits[action] = 1
        else:
            self.bits[action] = 0

        # アクションの実行結果を評価
        if self.check_is_completed():
            # 完了した場合、ボーナス報酬を与え、エピソード終了
            terminated = True
            reward += 10.0
        elif self.step_count >= self.n_max_steps:
            # ステップ上限に達した場合、エピソード終了
            terminated = True

        obs = self.bits.astype(np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        """
        現在の状態を表示
        """
        ones_count = np.sum(self.bits)
        print(f"Step {self.step_count}: [{','.join(map(str, self.bits))}]")
        print(f"Ones: {ones_count}/{self.n_bits} ({ones_count/self.n_bits:.2%})")

    def get_progress(self):
        """
        完了した割合を返す（1の数の割合）

        Returns:
            float: 完了した割合 (0~100)
        """
        return np.sum(self.bits) / self.n_bits * 100

    def check_is_completed(self):
        """
        完了したかチェック（すべて1になったか）

        Returns:
            bool: 完了したかどうか
        """
        return np.all(self.bits == 1)
