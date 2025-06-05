import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from settings import Settings


class OneMaxPlotter:
    def __init__(self, log_dir: str) -> None:
        """
        コンストラクタ

        Args:
            log_dir (str): ログディレクトリ
        """
        self._log_dir = log_dir

        # ログディレクトリの作成
        os.makedirs(self._log_dir, exist_ok=True)

    def _load_train_data(self) -> pd.DataFrame:
        """
        学習のモニターデータを読み込む

        Returns:
            pd.DataFrame: 学習データ

        Raises:
            Exception: 学習データ読み込みエラー
        """
        train_file = os.path.join(self._log_dir, "train.monitor.csv")

        try:
            train_data = pd.read_csv(train_file, skiprows=1)
            print(f"学習データ読み込み成功: {len(train_data)}行")
            print(f"カラム: {train_data.columns.tolist()}")
            return train_data
        except Exception as e:
            print(f"学習データ読み込みエラー: {e}")
            raise e

    def _load_eval_data(self) -> pd.DataFrame | None:
        """
        評価用のモニターデータを読み込む

        Returns:
            pd.DataFrame | None: 評価データ

        Raises:
            Exception: 評価データ読み込みエラー
        """
        eval_file = os.path.join(self._log_dir, "eval.monitor.csv")

        try:
            eval_data = pd.read_csv(eval_file, skiprows=1)
            print(f"評価データ読み込み成功: {len(eval_data)}行")
            print(f"カラム: {eval_data.columns.tolist()}")
            return eval_data
        except Exception as e:
            print(f"評価データ読み込みエラー: {e}")
            return None

    def _plot_reward_curve(self, train_data: pd.DataFrame, eval_data: pd.DataFrame | None, n_eval_freq: int, n_eval_episodes: int) -> None:
        """
        報酬曲線をプロット

        Args:
            train_data (pd.DataFrame): 学習データ
            eval_data (pd.DataFrame): 評価データ
        """
        plt.subplot(2, 2, 1)

        episodes = range(len(train_data))
        rewards = train_data["r"].values.astype(float)

        # 生の報酬値をプロット
        plt.plot(episodes, rewards, "b-", alpha=0.3, label="Training rewards")

        # 移動平均の計算とプロット
        window_size = min(50, len(rewards))
        if len(rewards) >= window_size:
            y_smooth = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
            x_smooth = range(window_size - 1, len(rewards))
            plt.plot(x_smooth, y_smooth, "r-", label=f"Moving average ({window_size})")

        # 評価データの報酬プロット
        if eval_data is not None:
            # タイムステップの累積を計算（もしまだない場合）
            if "timesteps" not in train_data.columns:
                train_data["timesteps"] = train_data["l"].cumsum()

            # 各評価ポイントが何エピソード目に相当するかを計算
            n_evaluations = len(eval_data) // n_eval_episodes

            # 各評価ポイントのタイムステップとエピソード番号を計算
            eval_timesteps = [(i + 1) * n_eval_freq for i in range(n_evaluations)]

            # タイムステップからエピソード番号への変換
            eval_episodes = []
            for timestep in eval_timesteps:
                # そのタイムステップ以下のエピソード数をカウント
                episode = train_data[train_data["timesteps"] <= timestep].shape[0]
                eval_episodes.append(episode)

            # 各評価ポイントの平均報酬を計算
            eval_rewards_avg = []
            for i in range(n_evaluations):
                start_idx = i * n_eval_episodes
                end_idx = start_idx + n_eval_episodes
                avg_reward = eval_data["r"][start_idx:end_idx].mean()
                eval_rewards_avg.append(avg_reward)

            # 評価報酬をプロット
            plt.plot(eval_episodes, eval_rewards_avg, "g-", label="Evaluation rewards (avg)", linewidth=2)
            plt.scatter(eval_episodes, eval_rewards_avg, color="green", s=30)  # 評価ポイントを強調

        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Reward Curve")
        plt.legend()
        plt.grid(True)

    def _plot_reward_distribution(self, train_data: pd.DataFrame) -> None:
        """
        報酬分布をプロット

        Args:
            train_data (pd.DataFrame): 学習データ
        """
        plt.subplot(2, 2, 2)
        rewards = train_data["r"].values.astype(float)
        plt.hist(rewards, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.grid(True)

    def _plot_episode_length(self, train_data: pd.DataFrame) -> None:
        """
        エピソード長の推移をプロット

        Args:
            monitor_data (pd.DataFrame): メイン学習データ
        """
        plt.subplot(2, 2, 3)
        episodes = range(len(train_data))
        episode_lengths = train_data["l"].values.astype(float)
        plt.plot(episodes, episode_lengths, "b-", alpha=0.7)
        plt.xlabel("Episodes")
        plt.ylabel("Episode Length")
        plt.title("Episode Length Curve")
        plt.grid(True)

    def _print_training_summary(self, train_data: pd.DataFrame) -> None:
        """
        学習の進捗サマリーを出力

        Args:
            train_data (pd.DataFrame): 学習データ
        """
        print("\n--- One Max 学習の進捗サマリー ---")
        print(f"総エピソード数: {len(train_data)}")
        print(f"平均エピソード長: {train_data['l'].mean():.2f}ステップ")
        print(f"初期の平均報酬 (最初の10エピソード): {train_data['r'][:10].mean():.2f}")
        print(f"最終の平均報酬 (最後の10エピソード): {train_data['r'][-10:].mean():.2f}")
        print(f"最高報酬: {train_data['r'].max():.2f}")
        print(f"最低報酬: {train_data['r'].min():.2f}")

        initial_reward = train_data["r"][:10].mean()
        final_reward = train_data["r"][-10:].mean()
        if initial_reward != 0:
            improvement = (final_reward - initial_reward) / abs(initial_reward) * 100
            print(f"報酬の改善率: {improvement:.2f}%")

        # One Max特有の分析
        high_reward_episodes = len(train_data[train_data["r"] > train_data["r"].quantile(0.9)])
        print(f"高報酬エピソード数 (上位10%): {high_reward_episodes}")

        # 短いエピソードの分析（効率的な解決）
        short_episodes = len(train_data[train_data["l"] <= train_data["l"].quantile(0.1)])
        print(f"短時間解決エピソード数 (下位10%): {short_episodes}")

    def _print_eval_summary(self, eval_data: pd.DataFrame) -> None:
        """
        評価結果サマリーを出力

        Args:
            eval_data (pd.DataFrame): 評価データ
        """
        print("\n--- 評価結果サマリー ---")
        print(f"評価エピソード数: {len(eval_data)}")
        print(f"評価時の平均報酬: {eval_data['r'].mean():.2f}")
        print(f"評価時の最高報酬: {eval_data['r'].max():.2f}")
        print(f"評価時の平均エピソード長: {eval_data['l'].mean():.2f}")

    def plot(self, n_eval_freq: int, n_eval_episodes: int, save_plot: bool = True, show_plot: bool = True) -> None:
        """
        学習曲線をプロットして保存・表示

        Args:
            save_plot (bool, optional): プロットを保存するか
            show_plot (bool, optional): プロットを表示するか
        """
        # データの読み込み
        train_data = self._load_train_data()
        eval_data = self._load_eval_data()

        # 学習曲線の可視化
        plt.figure(figsize=(15, 10))

        # 各種プロットの作成
        self._plot_reward_curve(
            train_data,
            eval_data,
            n_eval_freq,
            n_eval_episodes,
        )
        self._plot_reward_distribution(train_data)
        self._plot_episode_length(train_data)

        plt.tight_layout()

        # プロットの保存
        if save_plot:
            plot_path = os.path.join(self._log_dir, "learning_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"学習曲線を {plot_path} に保存しました")

        # プロットの表示
        if show_plot:
            plt.show()

        # サマリーの出力
        self._print_training_summary(train_data)
        if eval_data is not None:
            self._print_eval_summary(eval_data)


if __name__ == "__main__":
    # プロッター実行
    plotter = OneMaxPlotter(log_dir=Settings.LOG_DIR)
    plotter.plot(
        n_eval_freq=Settings.N_EVAL_FREQ,
        n_eval_episodes=Settings.N_EVAL_EPISODES,
        save_plot=True,
        show_plot=True,
    )
