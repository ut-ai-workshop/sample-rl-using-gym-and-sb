import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from mask_env import MaskOneMaxEnv
from settings import Settings


class OneMaxTrainer:
    def __init__(self, log_dir: str) -> None:
        """
        コンストラクタ

        Args:
            log_dir (str): ログディレクトリ
        """
        self._log_dir = log_dir

        # ログディレクトリの作成
        os.makedirs(self._log_dir, exist_ok=True)

    def _create_vec_env(self, log_filename: str, env: MaskOneMaxEnv) -> VecEnv:
        """
        ベクトル化環境の作成

        Args:
            log_filename (str): ログファイル名
            env (MaskOneMaxEnv): マスク環境

        Returns:
            VecEnv: ベクトル化環境
        """
        mask_env = ActionMasker(env, env.action_mask_func)
        monitor = Monitor(mask_env, log_filename)
        vec_env = DummyVecEnv([lambda: monitor])
        return vec_env

    def _create_callback(self, eval_env: MaskOneMaxEnv, n_eval_freq: int, n_eval_episodes: int) -> MaskableEvalCallback:
        """
        評価コールバックの作成

        Args:
            eval_env (MaskOneMaxEnv): 評価環境
            n_eval_freq (int): 評価頻度
            n_eval_episodes (int): 評価エピソード数

        Returns:
            MaskableEvalCallback: 評価コールバック
        """
        eval_callback = MaskableEvalCallback(
            self._create_vec_env(log_filename=f"{self._log_dir}/eval", env=eval_env),
            best_model_save_path=f"{self._log_dir}/best_model",
            log_path=f"{self._log_dir}/results",
            eval_freq=n_eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        return eval_callback

    def _create_model(self, train_env: MaskOneMaxEnv) -> MaskablePPO:
        """
        モデルの作成

        Args:
            train_env (MaskOneMaxEnv): 訓練環境

        Returns:
            MaskablePPO: モデル
        """
        model = MaskablePPO(
            "MlpPolicy",
            self._create_vec_env(log_filename=f"{self._log_dir}/train", env=train_env),
            n_steps=1024,
            batch_size=128,
            clip_range=0.2,
            learning_rate=3e-4,
            ent_coef=0.01,
            verbose=1,
            device="auto",
        )
        return model

    def train(self, train_env: MaskOneMaxEnv, eval_env: MaskOneMaxEnv, train_steps: int, n_eval_freq: int, n_eval_episodes: int) -> None:
        """
        訓練

        Args:
            train_env (MaskOneMaxEnv): 訓練環境
            eval_env (MaskOneMaxEnv): 評価環境
            train_steps (int): 訓練ステップ数
        """
        print("Starting training...")
        eval_callback = self._create_callback(
            eval_env=eval_env,
            n_eval_freq=n_eval_freq,
            n_eval_episodes=n_eval_episodes,
        )
        model = self._create_model(train_env=train_env)
        model.learn(
            total_timesteps=train_steps,
            progress_bar=True,
            callback=eval_callback,
        )
        model.save(f"{self._log_dir}/final_model/ppo_onemax")
        print("Training completed!")


if __name__ == "__main__":
    # 環境作成
    train_env = MaskOneMaxEnv(
        n_bits=Settings.N_BITS,
        initial_ones_ratio=Settings.INITIAL_ONES_RATIO,
        n_max_steps=Settings.N_MAX_STEPS,
        enable_mask=Settings.ENABLE_MASK,
    )
    eval_env = MaskOneMaxEnv(
        n_bits=Settings.N_BITS,
        initial_ones_ratio=Settings.INITIAL_ONES_RATIO,
        n_max_steps=Settings.N_MAX_STEPS,
        enable_mask=Settings.ENABLE_MASK,
    )

    # 訓練実行
    trainer = OneMaxTrainer(log_dir=Settings.LOG_DIR)
    trainer.train(
        train_env=train_env,
        eval_env=eval_env,
        train_steps=Settings.N_TRAIN_STEPS,
        n_eval_freq=Settings.N_EVAL_FREQ,
        n_eval_episodes=Settings.N_EVAL_EPISODES,
    )
