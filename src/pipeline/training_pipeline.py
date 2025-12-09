# src/pipelines/training_pipeline.py

import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_training_pipeline():
    """
    Ejecuta TODO el flujo end-to-end:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """
    try:
        logging.info("====== Iniciando Training Pipeline ======")

        # =====================================================
        # 1) DATA INGESTION
        # =====================================================
        logging.info("Paso 1: Data Ingestion...")

        ingestion = DataIngestion()
        train_path, val_path, test_path = ingestion.initiate_data_ingestion()

        logging.info(
            "Archivos generados -> %s, %s, %s",
            train_path,
            val_path,
            test_path,
        )

        # =====================================================
        # 2) DATA TRANSFORMATION
        # =====================================================
        logging.info("Paso 2: Data Transformation...")

        transformer = DataTransformation()
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
        ) = transformer.initiate_data_transformation()

        logging.info("Transformación completada. Arrays generados correctamente.")

        # =====================================================
        # 3) MODEL TRAINING
        # =====================================================
        logging.info("Paso 3: Model Training...")

        trainer = ModelTrainer()
        metrics = trainer.initiate_model_trainer(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
        )

        logging.info("Entrenamiento completado correctamente.")
        logging.info("Métricas finales: %s", metrics)

        logging.info("====== Training Pipeline Finalizado ======")
        return metrics

    except Exception as e:
        logging.error("Error durante la ejecución del pipeline", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        run_training_pipeline()
    except Exception as e:
        raise CustomException(e, sys)
