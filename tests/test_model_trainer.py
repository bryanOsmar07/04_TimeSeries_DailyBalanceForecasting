# tests/test_model_trainer.py

from pathlib import Path

import numpy as np

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


def test_model_trainer_basic(tmp_path):
    """
    Prueba básica del ModelTrainer sobre un dataset simulado.
    Solo verifica que la función corre y devuelve las claves esperadas,
    guardando artifacts SOLO en un directorio temporal.
    """
    # Dataset simulado pequeño
    n_features = 5
    X_train = np.random.rand(50, n_features)
    y_train = np.random.rand(50)

    X_val = np.random.rand(20, n_features)
    y_val = np.random.rand(20)

    X_test = np.random.rand(20, n_features)
    y_test = np.random.rand(20)

    # Nombres de features dummy
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    # ---------- Config de artifacts en tmp_path ----------
    artifacts_dir = tmp_path / "artifacts" / "model_trainer"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelTrainerConfig(
        artifacts_dir=str(artifacts_dir),
        model_path=str(artifacts_dir / "model.pkl"),
        metrics_path=str(artifacts_dir / "metrics.csv"),
        feature_importance_path=str(artifacts_dir / "feature_importances.csv"),
    )

    trainer = ModelTrainer(config=cfg)
    metrics = trainer.initiate_model_trainer(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
    )

    # ---------- Checks sobre el diccionario de métricas ----------
    assert "TRAIN" in metrics
    assert "VAL" in metrics
    assert "TEST" in metrics

    for split in ["TRAIN", "VAL", "TEST"]:
        assert "MAE" in metrics[split]
        assert "RMSE" in metrics[split]
        assert "MAPE" in metrics[split]
        assert "R2" in metrics[split]

    # Y opcionalmente, que los archivos se hayan creado SOLO en tmp_path
    assert Path(cfg.model_path).exists()
    assert Path(cfg.metrics_path).exists()
    assert Path(cfg.feature_importance_path).exists()
