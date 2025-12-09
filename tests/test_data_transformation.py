# tests/test_data_transformation.py

import os
from pathlib import Path

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)


def test_data_transformation(tmp_path):
    """
    Prueba que DataTransformation genere features y splits
    usando SOLO un directorio temporal para artifacts.
    """

    # ---------- 1) Config temporal para DataIngestion ----------
    artifacts_ing = tmp_path / "artifacts" / "data_ingestion"
    artifacts_ing.mkdir(parents=True, exist_ok=True)

    ing_cfg = DataIngestionConfig(
        source_csv_path=os.path.join("data", "raw", "data_fake_serie.csv"),
        artifacts_dir=str(artifacts_ing),
        raw_data_path=str(artifacts_ing / "raw.csv"),
        train_data_path=str(artifacts_ing / "train.csv"),
        val_data_path=str(artifacts_ing / "val.csv"),
        test_data_path=str(artifacts_ing / "test.csv"),
    )

    ingestion = DataIngestion(config=ing_cfg)
    ingestion.initiate_data_ingestion()

    # ---------- 2) Config temporal para DataTransformation ----------
    artifacts_tr = tmp_path / "artifacts" / "data_transformation"
    artifacts_tr.mkdir(parents=True, exist_ok=True)

    # OJO: usamos los nombres reales del dataclass:
    # raw_data_path, transformed_dir, train_output_path, val_output_path,
    # test_output_path, feature_cols_path
    tr_cfg = DataTransformationConfig(
        raw_data_path=str(ing_cfg.raw_data_path),
        transformed_dir=str(artifacts_tr),
        train_output_path=str(artifacts_tr / "train_transformed.csv"),
        val_output_path=str(artifacts_tr / "val_transformed.csv"),
        test_output_path=str(artifacts_tr / "test_transformed.csv"),
        # Usa .json si ya cambiaste el código; si sigue en .txt, deja .txt
        feature_cols_path=str(artifacts_tr / "feature_cols.txt"),
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

    # ---------- 3) Asserts: arrays y archivos ----------
    # Estructura de los arrays
    assert hasattr(X_train, "shape")
    assert X_train.shape[0] > 0
    assert hasattr(X_val, "shape")
    assert hasattr(X_test, "shape")

    assert isinstance(feature_cols, (list, tuple))
    assert len(feature_cols) > 0

    # Archivos generados SOLO en el tmp_dir
    assert Path(tr_cfg.train_output_path).exists()
    assert Path(tr_cfg.val_output_path).exists()
    assert Path(tr_cfg.test_output_path).exists()
    assert Path(tr_cfg.feature_cols_path).exists()
