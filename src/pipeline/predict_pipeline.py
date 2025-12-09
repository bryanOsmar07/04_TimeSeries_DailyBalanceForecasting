# src/pipeline/predict_pipeline.py

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import joblib
import pandas as pd

from src.components.data_transformation import build_features_from_saldo
from src.exception import CustomException
from src.logger import logging


@dataclass
class PredictConfig:
    """
    Configuración de rutas para el pipeline de predicción.
    Ajusta estas rutas según tu proyecto.
    """

    model_path: str = os.path.join(
        "artifacts", "model_trainer", "catboost_saldo_model.pkl"
    )
    feature_cols_path: str = os.path.join(
        "artifacts", "data_transformation", "feature_cols.json"
    )
    raw_data_path: str = os.path.join("data", "raw", "data_fake_serie.csv")
    forecast_days: int = 15
    forecast_output_path: str = os.path.join(
        "artifacts", "forecast", "forecast_15_days.csv"
    )


class PredictPipeline:
    """
    Pipeline de predicción / forecast para la serie de saldos diarios.
    """

    def __init__(self, config: Optional[PredictConfig] = None) -> None:
        try:
            self.config = config or PredictConfig()

            logging.info("Cargando modelo desde %s", self.config.model_path)
            self.model = joblib.load(self.config.model_path)

            logging.info(
                "Cargando feature_cols desde %s",
                self.config.feature_cols_path,
            )
            with open(
                self.config.feature_cols_path,
                "r",
                encoding="utf-8",
            ) as f:
                self.feature_cols = json.load(f)

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Forza 'load_date' a datetime y ordena el DataFrame.
        """
        if "load_date" not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'load_date'.")
        if "saldo" not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'saldo'.")

        df = df.copy()
        df["load_date"] = pd.to_datetime(df["load_date"])
        df = df.sort_values("load_date").reset_index(drop=True)
        return df

    def _iterative_forecast(
        self,
        df_history_raw: pd.DataFrame,
        n_days: int,
    ) -> pd.DataFrame:
        """
        Forecast iterativo día a día, recalculando features en cada paso.

        Parameters
        ----------
        df_history_raw : DataFrame con columnas ['load_date', 'saldo'] (histórico real)
        n_days         : horizonte de días a pronosticar

        Returns
        -------
        DataFrame con columnas ['load_date', 'saldo_pred'] para los días futuros.
        """
        try:
            logging.info("Iniciando forecast iterativo de %d días...", n_days)

            # Aseguramos orden temporal y tipos correctos
            df_work = self._ensure_datetime(df_history_raw)

            preds = []

            for step in range(1, n_days + 1):
                # Recalcular TODAS las features con la historia actual
                df_feat = build_features_from_saldo(df_work)

                # Tomar solo la última fila (la más reciente con features completos)
                last_row = df_feat.iloc[-1]

                # Asegurarnos de que tomamos exactamente las mismas columnas
                # que se usaron en entrenamiento
                cols_to_use = [c for c in self.feature_cols if c in df_feat.columns]
                X_input = last_row[cols_to_use].values.reshape(1, -1)

                # Predecir saldo t+1
                pred_saldo_next = float(self.model.predict(X_input)[0])

                # Nueva fecha: último día + 1
                last_date = df_work["load_date"].iloc[-1]
                new_date = last_date + pd.Timedelta(days=1)

                new_row = {
                    "load_date": new_date,
                    "saldo": pred_saldo_next,
                }

                # Añadir la nueva observación al histórico
                df_work = pd.concat(
                    [df_work, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

                preds.append((new_date, pred_saldo_next))

            df_forecast = pd.DataFrame(preds, columns=["load_date", "saldo_pred"])
            logging.info("Forecast iterativo completado.")
            return df_forecast

        except Exception as e:
            raise CustomException(e, sys) from e

    def forecast_next_days(
        self,
        raw_data_path: Optional[str] = None,
        n_days: Optional[int] = None,
        save_to_disk: bool = True,
    ) -> pd.DataFrame:
        """
        Método principal para hacer forecast N días a partir del histórico.

        Parameters
        ----------
        raw_data_path : ruta opcional al CSV con columnas ['load_date', 'saldo'].
                        Si es None, usa self.config.raw_data_path.
        n_days: horizonte de forecast. Si es None, usa self.config.forecast_days.
        save_to_disk  : si True, guarda el resultado en forecast_output_path

        Returns
        -------
        DataFrame con columnas ['load_date', 'saldo_pred'].
        """
        try:
            csv_path = raw_data_path or self.config.raw_data_path
            horizon = n_days or self.config.forecast_days

            logging.info("Leyendo histórico desde: %s", csv_path)
            df_hist = pd.read_csv(csv_path)

            df_hist = self._ensure_datetime(df_hist)

            df_forecast = self._iterative_forecast(
                df_history_raw=df_hist,
                n_days=horizon,
            )

            if save_to_disk:
                os.makedirs(
                    os.path.dirname(self.config.forecast_output_path),
                    exist_ok=True,
                )

                df_forecast.to_csv(self.config.forecast_output_path, index=False)

                logging.info(
                    "Forecast guardado en: %s",
                    self.config.forecast_output_path,
                )

            return df_forecast

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    """
    Ejecución directa:
    python -m src.pipeline.predict_pipeline
    """
    try:
        pipeline = PredictPipeline()
        df_fc = pipeline.forecast_next_days()
        print("Forecast generado (primeras filas):")
        print(df_fc.head())
    except Exception as e:
        # Usamos CustomException para loguear bonito
        raise CustomException(e, sys) from e
