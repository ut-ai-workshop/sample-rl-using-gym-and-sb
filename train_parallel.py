import os
from typing import Callable

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from mask_env import MaskOneMaxEnv
from settings import Settings


class OneMaxParallelTrainer:
    def __init__(self, log_dir: str) -> None:
        """
        コンストラクタ

        Args:
            log_dir (str): ログディレクトリ
        """
        self._log_dir = log_dir

        # ログディレクトリの作成
        os.makedirs(self._log_dir, exist_ok=True)

    def _create_parallel_vec_env(self, n_envs: int, log_filename: str, env_func: Callable) -> VecMonitor:
        """
        並列ベクトル化環境の作成

        Args:
            n_envs (int): 並列環境数
            log_filename (str): ログファイル名
            n_bits (int): ビット数
            env_func (Callable): 環境作成関数

        Returns:
            VecMonitor: 並列ベクトル化環境
        """
        env_fns = [env_func for _ in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        vec_monitor = VecMonitor(vec_env, log_filename)
        return vec_monitor

    def _create_callback(self, eval_env: VecMonitor, n_eval_freq: int, n_eval_episodes: int, n_train_envs: int) -> MaskableEvalCallback:
        """
        評価コールバックの作成

        Args:
            eval_env (VecMonitor): 評価環境
            n_eval_freq (int): 評価頻度
            n_eval_episodes (int): 評価エピソード数
            n_train_envs (int): 訓練環境数

        Returns:
            MaskableEvalCallback: 評価コールバック
        """
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=f"{self._log_dir}/best_model",
            log_path=f"{self._log_dir}/results",
            eval_freq=n_eval_freq // n_train_envs,  # 並列数で割る
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        return eval_callback

    def _create_model(self, train_env: VecMonitor, n_train_envs: int) -> MaskablePPO:
        """
        モデルの作成

        Args:
            train_env (VecMonitor): 訓練環境

        Returns:
            MaskablePPO: モデル
        """
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            n_steps=1024 // n_train_envs,
            batch_size=128,
            clip_range=0.2,
            learning_rate=3e-4,
            ent_coef=0.01,
            verbose=1,
            device="auto",
        )
        return model

    def train(self, env_func: Callable, train_steps: int, n_eval_freq: int, n_eval_episodes: int, n_train_envs: int, n_eval_envs: int = 3) -> None:
        """
        並列訓練

        Args:
            n_bits (int): ビット数
            initial_ones_ratio (float): 初期状態での1の比率
            n_max_steps (int): 最大ステップ数
            enable_mask (bool): マスクを有効にするか
            train_steps (int): 訓練ステップ数
            n_eval_freq (int): 評価頻度
            n_eval_episodes (int): 評価エピソード数
            n_train_envs (int): 訓練環境数
            n_eval_envs (int): 評価環境数
        """
        print(f"Starting parallel training with {n_train_envs} environments...")

        # 訓練用並列環境の作成
        train_env = self._create_parallel_vec_env(
            n_envs=n_train_envs,
            log_filename=os.path.join(self._log_dir, "train"),
            env_func=env_func,
        )

        # 評価用並列環境の作成
        eval_env = self._create_parallel_vec_env(
            n_envs=n_eval_envs,
            log_filename=os.path.join(self._log_dir, "eval"),
            env_func=env_func,
        )

        # 評価コールバックの作成
        eval_callback = self._create_callback(
            eval_env=eval_env,
            n_eval_freq=n_eval_freq,
            n_eval_episodes=n_eval_episodes,
            n_train_envs=n_train_envs,
        )

        # モデルの作成
        model = self._create_model(train_env=train_env, n_train_envs=n_train_envs)

        # 学習開始
        print(f"Total timesteps: {train_steps}")
        model.learn(
            total_timesteps=train_steps,
            progress_bar=True,
            callback=eval_callback,
        )

        # モデルの保存
        model_path = f"{self._log_dir}/final_model/ppo_onemax"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        print("Parallel training completed!")


if __name__ == "__main__":

    def env_func():
        env = MaskOneMaxEnv(
            n_bits=Settings.N_BITS,
            initial_ones_ratio=Settings.INITIAL_ONES_RATIO,
            n_max_steps=Settings.N_MAX_STEPS,
            enable_mask=Settings.ENABLE_MASK,
        )
        return ActionMasker(env, env.action_mask_func)

    # 訓練実行
    trainer = OneMaxParallelTrainer(log_dir=Settings.LOG_DIR)
    trainer.train(
        env_func=env_func,
        train_steps=Settings.N_TRAIN_STEPS,
        n_eval_freq=Settings.N_EVAL_FREQ,
        n_eval_episodes=Settings.N_EVAL_EPISODES,
        n_train_envs=Settings.N_TRAIN_ENVS,
        n_eval_envs=Settings.N_EVAL_ENVS,
    )
