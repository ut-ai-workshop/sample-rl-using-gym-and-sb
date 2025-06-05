from copy import deepcopy

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Sudoku1dEnv(gym.Env):
    """
    1次元数独問題: n個の位置に1からnまでの数字を重複なく配置する
    行動: 各位置の数字を次の値にサイクリックに変更 (0→1→2→...→n→0)
    0は空白を意味する
    """

    def __init__(self, n_size: int, n_blanks: int, n_max_steps: int):
        """
        1次元数独環境の初期化

        Args:
            n_size (int): 数独のサイズ（1からn_sizeまでの数字を使用、0は空白）
            n_blanks (int): 初期状態で作る空白マスの数
            n_max_steps (int): 最大ステップ数
        """
        super().__init__()
        self.n_size = n_size
        self.n_max_steps = n_max_steps
        self.n_blanks = n_blanks

        self.step_count = 0
        self.numbers = self._create_puzzle()
        self.initial_numbers = deepcopy(self.numbers)

        # 行動空間: n_sizeの位置
        self.action_space = spaces.Discrete(n_size)
        # 観測空間: 0からn_sizeまでの整数値（0は空白）
        self.observation_space = spaces.Box(low=0, high=n_size, shape=(n_size,), dtype=np.int32)

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

        self.step_count = 0
        self.numbers = self._create_puzzle()
        self.initial_numbers = deepcopy(self.numbers)

        obs = self.numbers.copy()
        return obs, {}

    def step(self, action: int):
        """
        1ステップの実行

        Args:
            action (int): 実行するアクション（変更する位置）

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: 観測, 報酬, 終了, 切り捨て, 情報
        """
        reward = 0
        terminated = False
        truncated = False
        self.step_count += 1

        # アクションを実行: 指定位置の数字をサイクリックに変更 (0→1→...→n→0)
        self.numbers[action] = (self.numbers[action] + 1) % (self.n_size + 1)

        # アクションの実行結果を評価
        if self.check_is_completed():
            # 完了した場合、ボーナス報酬を与え、エピソード終了
            terminated = True
            reward += 10.0
        elif self.step_count >= self.n_max_steps:
            # ステップ上限に達した場合、エピソード終了
            terminated = True

        obs = self.numbers.copy()
        return obs, reward, terminated, truncated, {}

    def render(self):
        """
        現在の状態を表示
        """
        unique_count = self._count_unique_numbers()
        duplicates = self._get_duplicates()
        missing = self._get_missing_numbers()
        print(f"Step {self.step_count}: [{','.join(map(str, self.numbers))}]")
        print(f"Unique numbers: {unique_count}/{self.n_size}")
        if duplicates:
            print(f"Duplicates: {duplicates}")
            print(f"Missing: {missing}")

    def get_progress(self):
        """
        完了した割合を返す（重複のない数字の割合）

        Returns:
            float: 完了した割合 (0~100)
        """
        return self._count_unique_numbers() / self.n_size * 100

    def check_is_completed(self):
        """
        完了したかチェック（1からn_sizeまでの数字が重複なく配置されているか）

        Returns:
            bool: 完了したかどうか
        """
        return len(np.unique(self.numbers)) == self.n_size and np.min(self.numbers) == 1 and np.max(self.numbers) == self.n_size

    def _count_unique_numbers(self):
        """
        重複のない数字の数を数える

        Returns:
            int: 重複のない数字の数
        """
        return len(np.unique(self.numbers))

    def _get_duplicates(self):
        """
        重複している数字のリストを取得

        Returns:
            list: 重複している数字のリスト
        """
        unique, counts = np.unique(self.numbers, return_counts=True)
        duplicates = unique[counts > 1]
        return duplicates.tolist()

    def _get_missing_numbers(self):
        """
        不足している数字のリストを取得

        Returns:
            list: 1からn_sizeの中で配置されていない数字のリスト
        """
        all_numbers = set(range(1, self.n_size + 1))
        current_numbers = set(self.numbers)
        missing = list(all_numbers - current_numbers)
        return sorted(missing)

    def _create_puzzle(self):
        """
        数独問題を作成
        """
        puzzle = self._create_solution()

        # 指定された数の位置をランダムに選んで空白（0）にする
        if self.n_blanks > 0:
            blank_positions = np.random.choice(self.n_size, size=self.n_blanks, replace=False)
            puzzle[blank_positions] = 0
        return puzzle

    def _create_solution(self):
        """
        正解を作成
        """
        # まず完全な解（1からn_sizeまでの順列）を作成
        solution = np.arange(1, self.n_size + 1, dtype=np.int32)
        # ランダムにシャッフル
        np.random.shuffle(solution)
        return solution
