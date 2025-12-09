# src/components/data_ingestion.py

import os
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Rutas por defecto de los archivos generados en la etapa de ingestión.
    """

    # Archivo de origen
    source_csv_path: str = os.path.join("data", "raw", "data_fake_serie.csv")

    # Carpeta de artifacts
    artifacts_dir: str = os.path.join("artifacts", "data_ingestion")

    # Archivo crudo (copia del .csv original)
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")

    # Splits temporales
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    val_data_path: str = os.path.join("artifacts", "data_ingestion", "val.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")


class DataIngestion:
    """
    - Lee el dataset original desde un .csv
    - Ordena por fecha (load_date)
    - Genera splits temporales: train (70%), val (15%), test (15%)
    - Guarda raw, train, val y test en la carpeta artifacts
    """

    def __init__(self, config: Optional[DataIngestionConfig] = None) -> None:
        """
        Si no se pasa config, usa la configuración por defecto.
        Esto permite que los tests usen rutas temporales sin pisar artifacts reales.
        """
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        source_csv_path: ruta al archivo .csv original con al menos:
            - load_date (fecha)
            - saldo (serie a modelar)

        Devuelve:
            (train_data_path, val_data_path, test_data_path)
        """
        logging.info("Iniciando proceso de Data Ingestion")

        try:
            source_csv_path = self.config.source_csv_path

            # ===========================
            # 1. Lectura del dataset
            # ===========================
            if not os.path.exists(source_csv_path):
                msg = f"Archivo de origen no encontrado: {source_csv_path}"
                logging.error(msg)
                raise FileNotFoundError(msg)

            logging.info(f"Leyendo dataset desde: {source_csv_path}")
            df = pd.read_csv(source_csv_path)

            # Aseguramos parseo de fecha (ajusta el nombre si es distinto)
            if "load_date" in df.columns:
                df["load_date"] = pd.to_datetime(df["load_date"])
            else:
                raise CustomException(
                    "La columna 'load_date' no existe en el dataset.", sys
                )

            # Orden temporal
            df = df.sort_values("load_date").reset_index(drop=True)
            logging.info(f"Dataset leído con shape: {df.shape}")

            # Creamos carpeta de artifacts si no existe
            os.makedirs(self.config.artifacts_dir, exist_ok=True)

            # Guardamos copia cruda
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Archivo raw guardado en: {self.config.raw_data_path}")

            # ===========================
            # 2. Split temporal
            # ===========================
            n = len(df)
            train_size = int(n * 0.70)
            val_size = int(n * 0.15)

            train_df = df.iloc[:train_size].copy()
            val_df = df.iloc[train_size : train_size + val_size].copy()
            test_df = df.iloc[train_size + val_size :].copy()

            logging.info(
                "Splits generados - "
                f"train: {train_df.shape}, "
                f"val: {val_df.shape}, "
                f"test: {test_df.shape}"
            )

            # ===========================
            # 3. Guardar splits
            # ===========================
            train_df.to_csv(self.config.train_data_path, index=False)
            val_df.to_csv(self.config.val_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("Archivos de train/val/test guardados correctamente")

            return (
                self.config.train_data_path,
                self.config.val_data_path,
                self.config.test_data_path,
            )

        except Exception as e:
            logging.error("Error durante la ingestión de datos", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    # Probar el data_ingestion
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    print(train_path, val_path, test_path)
