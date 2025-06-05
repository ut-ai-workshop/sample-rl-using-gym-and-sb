import numpy as np

from env import OneMaxEnv


class MaskOneMaxEnv(OneMaxEnv):
    """
    マスク付きOne Max環境
    すでに1のビットはフリップしない（1の数を減らさない）制約を追加
    """

    def __init__(self, n_bits: int, initial_ones_ratio: float, n_max_steps: int, enable_mask: bool = True):
        """
        マスク付きOne Max環境の初期化

        Args:
            n_bits (int): ビット数
            initial_ones_ratio (float): 初期状態での1の比率
            n_max_steps (int): 最大ステップ数
            enable_mask (bool): マスクを有効にするか
        """
        super().__init__(
            n_bits,
            initial_ones_ratio,
            n_max_steps,
        )
        self.enable_mask = enable_mask  # マスクを有効にするか

    def compute_action_mask(self) -> np.ndarray:
        """
        行動マスクを計算する内部メソッド

        Returns:
            np.ndarray: 行動マスク
        """
        # すべての行動を許可
        action_mask = np.ones(self.n_bits, dtype=bool)

        if self.enable_mask:
            # 1→0への変更を禁止
            action_mask[self.initial_bits == 1] = False

        return action_mask

    def step(self, action: int):
        """
        1ステップの実行

        Args:
            action (int): 実行するアクション

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: 観測, 報酬, 終了, 切り捨て, 情報
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # まだ終わっていない場合、有効な行動が残っているかチェック
        if not terminated and not truncated:
            action_mask = self.compute_action_mask()
            if not action_mask.any() and not self.check_is_completed():
                # 有効な行動がないが完了していない場合、終了（必要があればペナルティを与える）
                # reward = -10
                terminated = True

        return obs, reward, terminated, truncated, info

    @staticmethod
    def action_mask_func(env) -> np.ndarray:
        """
        行動マスク関数

        Args:
            env: 環境

        Returns:
            np.ndarray: 行動マスク
        """
        return env.compute_action_mask()
