# src/pipelines/training_pipeline.py

import sys
import logging

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig
)

from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig
)

from src.components.model_trainer import (
    ModelTrainer,
    ModelTrainerConfig
)


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

        ingestion = DataIngestion(config=DataIngestionConfig())
        train_path, val_path, test_path = ingestion.initiate_data_ingestion()

        logging.info(f"Archivos generados -> {train_path}, {val_path}, {test_path}")

        # =====================================================
        # 2) DATA TRANSFORMATION
        # =====================================================
        logging.info("Paso 2: Data Transformation...")

        transformer = DataTransformation(config=DataTransformationConfig())
        (
            X_train_path,
            y_train_path,
            X_val_path,
            y_val_path,
            X_test_path,
            y_test_path,
            feat_path
        ) = transformer.initiate_data_transformation()

        logging.info("Transformación completada. Archivos creados correctamente.")

        # =====================================================
        # 3) MODEL TRAINING
        # =====================================================
        logging.info("Paso 3: Model Training...")

        trainer = ModelTrainer(config=ModelTrainerConfig())
        metrics = trainer.initiate_model_trainer()

        logging.info("Entrenamiento completado correctamente.")
        logging.info(f"Métricas finales: {metrics}")

        logging.info("====== Training Pipeline Finalizado ======")
        return metrics

    except Exception as e:
        logging.error("Error durante la ejecución del pipeline", exc_info=True)
        raise CustomException(e, sys)
