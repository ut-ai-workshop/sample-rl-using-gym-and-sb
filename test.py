import os
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from mask_env import MaskOneMaxEnv
from settings import Settings


class OneMaxTester:
    def __init__(self, log_dir: str) -> None:
        """
        コンストラクタ

        Args:
            log_dir (str): ログディレクトリ
        """
        self._log_dir = log_dir

        # ログディレクトリの作成
        os.makedirs(self._log_dir, exist_ok=True)

    def _create_vec_env(self, env: MaskOneMaxEnv) -> VecEnv:
        """
        ベクトル化環境の作成

        Args:
            env (MaskOneMaxEnv): マスク環境

        Returns:
            VecEnv: ベクトル化環境
        """
        mask_env = ActionMasker(env, env.action_mask_func)
        monitor = Monitor(mask_env)
        vec_env = DummyVecEnv([lambda: monitor])
        return vec_env

    def _load_model(self, vec_env: VecEnv, model_path: str = "") -> MaskablePPO:
        """
        学習済みモデルの読み込み

        Args:
            vec_env (VecEnv): ベクトル化環境
            model_path (str, optional): モデルファイルのパス

        Returns:
            MaskablePPO: 読み込まれたモデル

        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
        """
        if not model_path:
            model_path = os.path.join(self._log_dir, "best_model", "best_model.zip")

        try:
            model = MaskablePPO.load(model_path, env=vec_env)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}")
            print(f"Error: {e}")

            # バックアップとして直接保存されたモデルを試す
            try:
                backup_path = f"{self._log_dir}/final_model/ppo_onemax"
                model = MaskablePPO.load(backup_path, env=vec_env)
                print("Loaded backup model: ppo_onemax.zip")
                return model
            except Exception as backup_e:
                raise FileNotFoundError("No trained model found. Please run train.py first.")

    def run_episode(self, model: MaskablePPO, base_env: MaskOneMaxEnv, deterministic: bool = True, verbose: bool = True) -> dict[str, Any]:
        """
        1エピソードを実行して結果を返す

        Args:
            model (MaskablePPO): 学習済みモデル
            base_env (MaskOneMaxEnv): 環境
            deterministic (bool, optional): 決定的行動を取るか
            verbose (bool, optional): 詳細出力するか

        Returns:
            Dict[str, Any]: エピソード結果
        """
        obs, _ = base_env.reset()

        if verbose:
            print("Initial state:")
            base_env.render()
            print("")

        done = False
        step_count = 0
        total_reward = 0
        action_history = []
        reward_history = []

        while not done:
            mask = base_env.compute_action_mask()

            # 価値関数の予測（オプション）
            if isinstance(obs, dict):
                obs_tensor = {k: torch.as_tensor(v[None]).to(model.device) for k, v in obs.items()}
            else:
                obs_tensor = torch.as_tensor(obs[None]).to(model.device)

            v_pred = model.policy.predict_values(obs_tensor).item()

            # 行動を予測
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)

            # 環境を進める（actionをintに変換）
            action_int = int(action) if isinstance(action, np.ndarray) else action
            obs, reward, terminated, truncated, _ = base_env.step(action_int)

            total_reward += reward
            done = terminated or truncated
            step_count += 1

            action_history.append(action_int)
            reward_history.append(reward)

            if verbose:
                print(f"Step {step_count:2d}: Action {action_int:2d}, Reward {reward:+.1f}, " f"V(s) {v_pred:+.2f}")

        if verbose:
            print("\nFinal state:")
            base_env.render()
            print(f"\nEpisode summary:")
            print(f"  Total reward: {total_reward:.1f}")
            print(f"  Steps taken: {step_count}")
            print(f"  Final progress: {base_env.get_progress():.1f}%")
            print(f"  Completed: {base_env.check_is_completed()}")

        return {
            "total_reward": total_reward,
            "steps": step_count,
            "final_progress": base_env.get_progress(),
            "completed": base_env.check_is_completed(),
            "action_history": action_history,
            "reward_history": reward_history,
        }

    def test(self, test_env: MaskOneMaxEnv, num_tests: int = 100, model_path: str = "") -> None:
        """
        テストの実行

        Args:
            test_env (MaskOneMaxEnv): テスト用環境
            num_tests (int, optional): テスト回数
            model_path (str, optional): モデルファイルのパス
        """
        print(f"\n=== One Max Problem Solver ===")

        # テスト用環境の作成
        vec_env = self._create_vec_env(test_env)

        # 学習済みモデルの読み込み
        model = self._load_model(vec_env, model_path)

        # 複数回テストして性能を評価
        print(f"\n=== Multiple Test Runs ===")
        results = []

        for i in range(num_tests):
            print(f"\n\n--- Test Run {i+1}/{num_tests} ---")
            result = self.run_episode(
                model,
                test_env,
                deterministic=True,
                verbose=True,
            )
            results.append(result)
            print(f"\nResult: {result['final_progress']:.1f}% in {result['steps']} steps")

        # 統計サマリー
        self._print_summary(results, num_tests)

        # 環境をクローズ
        vec_env.close()

    def _print_summary(self, results: list[dict[str, Any]], num_tests: int) -> None:
        """
        統計サマリーの出力

        Args:
            results (list[dict[str, Any]]): テスト結果のリスト
            num_tests (int): テスト回数
        """
        final_progresses = [r["final_progress"] for r in results]
        steps_taken = [r["steps"] for r in results]
        completed_count = sum(r["completed"] for r in results)

        print(f"\n\n=== Performance Summary ({num_tests} runs) ===")
        print(f"Final progress: {np.mean(final_progresses):.1f} ± {np.std(final_progresses):.1f}%")
        print(f"Steps taken: {np.mean(steps_taken):.1f} ± {np.std(steps_taken):.1f}")
        print(f"Completed: {completed_count}/{num_tests} ({completed_count/num_tests:.1%})")
        print(f"Best progress: {max(final_progresses):.1f}%")
        print(f"Worst progress: {min(final_progresses):.1f}%")
        print("\n")


if __name__ == "__main__":
    # 環境設定
    test_env = MaskOneMaxEnv(
        n_bits=Settings.N_BITS,
        initial_ones_ratio=Settings.INITIAL_ONES_RATIO,
        n_max_steps=Settings.N_MAX_STEPS,
        enable_mask=Settings.ENABLE_MASK,
    )

    # テスト実行
    tester = OneMaxTester(log_dir=Settings.LOG_DIR)
    tester.test(
        test_env=test_env,
        num_tests=Settings.N_TEST_EPISODES,
    )
