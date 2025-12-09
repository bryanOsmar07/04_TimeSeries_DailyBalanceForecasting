# tests/test_model_trainer.py

import numpy as np
from src.components.model_trainer import ModelTrainer


def test_model_trainer_basic():
    """
    Prueba básica del ModelTrainer sobre un dataset simulado.
    Solo verifica que la función corre y devuelve las claves esperadas.
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

    trainer = ModelTrainer()  # Usa config por defecto
    metrics = trainer.initiate_model_trainer(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_cols,
    )

    assert "TRAIN" in metrics
    assert "VAL" in metrics
    assert "TEST" in metrics

    for split in ["TRAIN", "VAL", "TEST"]:
        assert "MAE" in metrics[split]
        assert "RMSE" in metrics[split]
        assert "MAPE" in metrics[split]
        assert "R2" in metrics[split]
