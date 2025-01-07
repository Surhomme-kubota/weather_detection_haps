from pathlib import Path

# ベースパスの設定
BASEPATH = Path(__file__).resolve().parent.parent.parent

# トレーニング条件
TRAIN_CONDITIONS = {
    "train_batch_size": 8,
    "test_batch_size": 8,
    "evaluation_batch_size": 1,
}

# モデルの設定
MODEL_SETTINGS = {
    "num_classes": 2,
    "label_smoothing": 0.1,
    "learning_rate": 0.01,
    "momentum": 0.1,
    "t_max": 10,
    "epochs": 10
}

# モデルの保存パス
MODEL_SAVE_DIR = BASEPATH / 'models' / 'rain'
MODEL_SAVE_PATHS = {
    "default_model": MODEL_SAVE_DIR / 'best_model.pth',
    "model1": MODEL_SAVE_DIR / 'best_model1.pth',
    "model2": MODEL_SAVE_DIR / 'best_model2.pth',
}

# パスを文字列に変換
MODEL_SAVE_PATHS_STR = {key: str(path) for key, path in MODEL_SAVE_PATHS.items()}