# tests/test_predict_pipeline.py

import numpy as np
import pandas as pd

import src.pipeline.predict_pipeline as pp
from src.pipeline.predict_pipeline import PredictPipeline


def test_predict_pipeline(tmp_path, monkeypatch):
    """
    Verifica que el forecast funcione sin fallar,
    usando un modelo dummy y feature_cols simuladas,
    SIN depender de artifacts reales en disco.

    Usamos un histórico suficientemente largo para que
    build_features_from_saldo no se quede sin filas.
    """

    # 1) Modelo dummy que emula el método .predict del CatBoost
    class DummyModel:
        def predict(self, X):
            # Devuelve un vector de ceros con la misma cantidad de filas
            return np.zeros(X.shape[0])

    # 2) Reemplazamos el __init__ de PredictPipeline para que:
    #    - No lea archivos .pkl ni .json reales
    #    - Inyecte DummyModel y feature_cols = ["saldo"]
    def fake_init(self, config=None):
        self.config = config or pp.PredictConfig()
        self.model = DummyModel()
        # Usamos una feature que sabemos que existe después del feature engineering.
        # build_features_from_saldo conserva la columna "saldo".
        self.feature_cols = ["saldo"]

    # Monkeypatch del __init__ de la clase
    monkeypatch.setattr(PredictPipeline, "__init__", fake_init)

    # 3) Generamos un histórico sintético suficientemente largo (p.ej. 120 días)
    n_days_hist = 120
    dates = pd.date_range("2024-01-01", periods=n_days_hist, freq="D")
    saldo = np.linspace(1000, 2000, n_days_hist)  # tendencia suave

    df_hist = pd.DataFrame(
        {
            "load_date": dates,
            "saldo": saldo,
        }
    )

    # Lo guardamos en un CSV temporal
    raw_csv_path = tmp_path / "raw_predict.csv"
    df_hist.to_csv(raw_csv_path, index=False)

    # 4) Creamos el pipeline (usará fake_init) y ejecutamos forecast_next_days
    pipeline = PredictPipeline()
    df_fc = pipeline.forecast_next_days(
        raw_data_path=str(raw_csv_path),
        n_days=5,
        save_to_disk=False,
    )

    # 5) Aserciones básicas: que no falle y tenga el formato esperado
    assert df_fc is not None
    assert len(df_fc) == 5
    assert "load_date" in df_fc.columns
    assert "saldo_pred" in df_fc.columns
