# tests/test_data_transformation.py

from pathlib import Path

import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)


def test_data_transformation(tmp_path):
    """
    Prueba que DataTransformation genere features y splits
    usando SOLO un dataset sintético y artifacts en tmp_path.
    """

    # ---------- 1) Crear CSV sintético ----------
    df = pd.DataFrame(
        {
            "load_date": pd.date_range("2024-01-01", periods=400, freq="D"),
            "saldo": range(400),
        }
    )
    source_csv = tmp_path / "data_fake_serie.csv"
    df.to_csv(source_csv, index=False)

    # ---------- 2) Config temporal para DataIngestion ----------
    artifacts_ing = tmp_path / "artifacts" / "data_ingestion"
    artifacts_ing.mkdir(parents=True, exist_ok=True)

    ing_cfg = DataIngestionConfig(
        source_csv_path=str(source_csv),
        artifacts_dir=str(artifacts_ing),
        raw_data_path=str(artifacts_ing / "raw.csv"),
        train_data_path=str(artifacts_ing / "train.csv"),
        val_data_path=str(artifacts_ing / "val.csv"),
        test_data_path=str(artifacts_ing / "test.csv"),
    )

    ingestion = DataIngestion(config=ing_cfg)
    ingestion.initiate_data_ingestion()

    # ---------- 3) Config temporal para DataTransformation ----------
    artifacts_tr = tmp_path / "artifacts" / "data_transformation"
    artifacts_tr.mkdir(parents=True, exist_ok=True)

    tr_cfg = DataTransformationConfig(
        raw_data_path=str(ing_cfg.raw_data_path),
        transformed_dir=str(artifacts_tr),
        train_output_path=str(artifacts_tr / "train_transformed.csv"),
        val_output_path=str(artifacts_tr / "val_transformed.csv"),
        test_output_path=str(artifacts_tr / "test_transformed.csv"),
        # usa .json o .txt según tengas en tu config real
        feature_cols_path=str(artifacts_tr / "feature_cols.json"),
    )

    transformer = DataTransformation(config=tr_cfg)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
    ) = transformer.initiate_data_transformation()

    # ---------- 4) Asserts básicos ----------
    assert hasattr(X_train, "shape")
    assert X_train.shape[0] > 0
    assert hasattr(X_val, "shape")
    assert hasattr(X_test, "shape")

    assert isinstance(feature_cols, (list, tuple))
    assert len(feature_cols) > 0

    # Archivos generados SOLO en tmp_path
    assert Path(tr_cfg.train_output_path).exists()
    assert Path(tr_cfg.val_output_path).exists()
    assert Path(tr_cfg.test_output_path).exists()
    assert Path(tr_cfg.feature_cols_path).exists()
