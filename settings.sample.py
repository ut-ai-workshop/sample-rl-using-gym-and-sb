class Settings:
    # ログディレクトリ
    LOG_DIR = "logs"

    # 環境設定
    N_SIZE = 4  # 数独のサイズ（1からn_sizeまでの数字を使用、0は空白）
    ENABLE_MASK = False  # True: マスクを有効にする, False: マスクを無効にする

    # 訓練設定
    N_TRAIN_STEPS = 100_000
    N_MAX_STEPS = N_SIZE * 20  # 1エピソードの最大ステップ数
    N_TRAIN_ENVS = 16  # 訓練用環境数
    N_EVAL_ENVS = 8  # 評価用環境数
    N_EVAL_FREQ = 2000
    N_EVAL_EPISODES = 5

    # 評価設定
    N_TEST_EPISODES = 10
