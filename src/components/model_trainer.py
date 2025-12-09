# src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    """
    Rutas de salida para el modelo entrenado y sus artefactos.
    """
    artifacts_dir: str = os.path.join("artifacts", "model_trainer")

    model_path: str = os.path.join(
        artifacts_dir, "catboost_saldo_model.pkl"
    )
    metrics_path: str = os.path.join(
        artifacts_dir, "metrics_catboost_saldo.csv"
    )
    feature_importance_path: str = os.path.join(
        artifacts_dir, "feature_importances_catboost_saldo.csv"
    )


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas de regresión básicas.
    """
    mae = mean_absolute_error(y_true, y_pred)
    #rmse = mean_squared_error(y_true, y_pred, squared=False)
    rmse = root_mean_squared_error(y_true, y_pred)

    # Evitamos divisiones por cero en MAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(
            np.abs((y_true - y_pred) / y_true)
        ) * 100.0
    if np.isinf(mape) or np.isnan(mape):
        mape = np.nan

    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def _build_model(self) -> CatBoostRegressor:
        """
        Devuelve una instancia de CatBoostRegressor con los
        hiperparámetros finos que encontraste en los notebooks.
        """
        model = CatBoostRegressor(
            loss_function="RMSE",
            depth=4,
            learning_rate=0.03,
            iterations=700,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
        )
        return model

    def initiate_model_trainer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_cols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Entrena el modelo CatBoost para el saldo:

        1. Entrena sobre TRAIN.
        2. Evalúa en Train / Val / Test.
        3. Reentrena modelo final con TRAIN + VAL.
        4. Guarda el modelo y las métricas.

        Devuelve un dict con las métricas para cada split.
        """
        logging.info("Iniciando Model Trainer para CatBoost (saldo)...")

        try:
            os.makedirs(self.config.artifacts_dir, exist_ok=True)

            # ============================
            # 1) Entrenamiento en TRAIN
            # ============================
            model = self._build_model()
            logging.info("Entrenando CatBoost solo con TRAIN...")
            model.fit(X_train, y_train)

            # ============================
            # 2) Predicciones y métricas
            # ============================
            logging.info("Calculando métricas en Train/Val/Test...")

            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)

            metrics_train = _compute_metrics(y_train, pred_train)
            metrics_val = _compute_metrics(y_val, pred_val)
            metrics_test = _compute_metrics(y_test, pred_test)

            # Armamos DataFrame de métricas
            df_metrics = pd.DataFrame(
                {
                    "TRAIN": metrics_train,
                    "VAL": metrics_val,
                    "TEST": metrics_test,
                }
            ).T

            df_metrics["modelo"] = "CatBoost tuned – saldo"

            df_metrics.to_csv(self.config.metrics_path, index=True)
            logging.info(
                "Métricas guardadas en %s", self.config.metrics_path
            )

            # ============================
            # 3) Feature importance
            # ============================
            logging.info("Calculando importancias de variables...")

            feature_importances = model.get_feature_importance()
            df_importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": feature_importances,
                }
            ).sort_values("importance", ascending=False)

            df_importance.to_csv(
                self.config.feature_importance_path, index=False
            )
            logging.info(
                "Importancias de variables guardadas en %s",
                self.config.feature_importance_path,
            )

            # ============================
            # 4) Reentrenar modelo final
            # ============================
            logging.info("Reentrenando CatBoost con TRAIN + VAL...")

            X_train_full = np.vstack([X_train, X_val])
            y_train_full = np.concatenate([y_train, y_val])

            final_model = self._build_model()
            final_model.fit(X_train_full, y_train_full)

            joblib.dump(final_model, self.config.model_path)
            logging.info(
                "Modelo final guardado en %s", self.config.model_path
            )

            # ============================
            # 5) Devolver resumen
            # ============================
            metrics_dict: Dict[str, Dict[str, float]] = {
                "TRAIN": metrics_train,
                "VAL": metrics_val,
                "TEST": metrics_test,
            }

            logging.info("Entrenamiento de CatBoost completado.")
            return metrics_dict

        except Exception as e:
            raise CustomException(e, sys)
