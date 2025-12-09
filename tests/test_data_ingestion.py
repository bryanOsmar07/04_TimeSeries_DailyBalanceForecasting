# tests/test_data_ingestion.py
from pathlib import Path

import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig


def test_data_ingestion(tmp_path):
    """
    Prueba que DataIngestion ejecute sin errores y genere
    archivos de train/val/test, usando un CSV sintético en tmp_path.
    """

    # 1) Creamos un CSV sintético con las columnas esperadas
    df = pd.DataFrame(
        {
            "load_date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "saldo": range(30),
        }
    )

    source_csv = tmp_path / "data_fake_serie.csv"
    df.to_csv(source_csv, index=False)

    # 2) Definimos rutas de artifacts SOLO dentro de tmp_path
    artifacts_dir = tmp_path / "artifacts" / "data_ingestion"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataIngestionConfig(
        source_csv_path=str(source_csv),
        artifacts_dir=str(artifacts_dir),
        raw_data_path=str(artifacts_dir / "raw.csv"),
        train_data_path=str(artifacts_dir / "train.csv"),
        val_data_path=str(artifacts_dir / "val.csv"),
        test_data_path=str(artifacts_dir / "test.csv"),
    )

    ingestion = DataIngestion(config=cfg)
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()

    # 3) Asserts: que se hayan creado los archivos
    assert Path(train_path).exists()
    assert Path(val_path).exists()
    assert Path(test_path).exists()
