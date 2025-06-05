class Settings:
    # ログディレクトリ
    LOG_DIR = "logs"

    # 環境設定
    N_BITS = 4  # ビット数（問題の難易度を調整）
    INITIAL_ONES_RATIO = 0.3  # 初期状態での1の比率
    ENABLE_MASK = False  # True: マスクを有効にする, False: マスクを無効にする

    # 訓練設定
    N_TRAIN_STEPS = 10_000
    N_MAX_STEPS = N_BITS * 2  # 1エピソードの最大ステップ数
    N_TRAIN_ENVS = 32  # 訓練用環境数
    N_EVAL_ENVS = 8  # 評価用環境数
    N_EVAL_FREQ = 2000
    N_EVAL_EPISODES = 5

    # 評価設定
    N_TEST_EPISODES = 10
