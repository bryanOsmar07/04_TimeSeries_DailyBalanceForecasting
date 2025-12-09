import json
import os
import sys
from typing import List

import joblib
import pandas as pd

from src.components.data_transformation import build_features_from_saldo
from src.exception import CustomException
from src.logger import logging

# Rutas “fijas” de artefactos (las mismas que vienes usando)
RAW_PATH = os.path.join("artifacts", "data_ingestion", "raw.csv")
FEATURE_COLS_PATH = os.path.join(
    "artifacts",
    "data_transformation",
    "feature_cols.json",
)
MODEL_PATH = os.path.join("artifacts", "model_trainer", "catboost_saldo_model.pkl")
FORECAST_DIR = os.path.join("artifacts", "forecast")


def forecast_next_n_days_catboost(
    model,
    df_history_raw: pd.DataFrame,
    feature_cols: List[str],
    n_days: int = 15,
) -> pd.DataFrame:
    """
    Genera un forecast multi-step autoregresivo para los próximos n_days
    usando el modelo CatBoost entrenado sobre SALDO.

    - model: modelo CatBoost ya entrenado (saldo).
    - df_history_raw: DataFrame con columnas ['load_date', 'saldo'].
    - feature_cols: lista de columnas usadas para entrenar el modelo.
    - n_days: horizonte de pronóstico.
    """

    df_work = df_history_raw.copy().sort_values("load_date").reset_index(drop=True)
    preds = []

    for step in range(1, n_days + 1):
        # Recalcular TODAS las features con la historia + predicciones previas
        df_feat = build_features_from_saldo(df_work)

        # Tomamos la última fila (la más reciente)
        last_row = df_feat.iloc[-1]
        X_input = last_row[feature_cols].values.reshape(1, -1)

        # Predecimos saldo t+1
        pred_saldo_next = model.predict(X_input)[0]

        # Fecha futura
        last_date = df_work["load_date"].iloc[-1]
        new_date = last_date + pd.Timedelta(days=1)

        # Añadimos nueva observación a la historia
        new_row = {"load_date": new_date, "saldo": pred_saldo_next}
        df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)

        preds.append((new_date, pred_saldo_next))

    df_forecast = pd.DataFrame(preds, columns=["load_date", "saldo_pred"])
    return df_forecast


def run_forecast_pipeline(n_days: int = 15) -> str:
    """
    Orquesta el forecast:

    1. Carga raw.csv (load_date, saldo).
    2. Carga feature_cols.json.
    3. Carga el modelo CatBoost entrenado.
    4. Genera el pronóstico para n_days.
    5. Guarda el forecast en artifacts/forecast/forecast_{n_days}d.csv.
    6. Devuelve la ruta del archivo generado.
    """
    try:
        logging.info("Iniciando forecast_pipeline para %s días...", n_days)

        # ============================
        # 1) Cargar histórico raw.csv
        # ============================
        if not os.path.exists(RAW_PATH):
            msg = (
                f"No se encontró el raw data en {RAW_PATH}. "
                "Ejecuta primero el training_pipeline."
            )
            raise FileNotFoundError(msg)

        df_raw = pd.read_csv(RAW_PATH)
        if "load_date" not in df_raw.columns or "saldo" not in df_raw.columns:
            raise ValueError("raw.csv debe contener 'load_date' y 'saldo'.")

        df_raw["load_date"] = pd.to_datetime(df_raw["load_date"])
        df_raw = df_raw.sort_values("load_date").reset_index(drop=True)

        # ============================
        # 2) Cargar feature_cols.json
        # ============================
        feature_cols = None

        if os.path.exists(FEATURE_COLS_PATH):
            # Camino normal: usar el JSON generado por training_pipeline
            with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)

            if not isinstance(feature_cols, list):
                raise ValueError(
                    f"El contenido de {FEATURE_COLS_PATH} debe ser lista de nombres."
                )
            logging.info(
                "feature_cols cargado desde %s con %d columnas.",
                FEATURE_COLS_PATH,
                len(feature_cols),
            )

        else:
            # 🔁 Plan B: reconstruir feature_cols a partir del raw
            logging.warning(
                "No se encontró %s. Reconstruyendo feature_cols a partir de raw...",
                FEATURE_COLS_PATH,
            )

            # Construimos todas las features con la función oficial
            df_feat_tmp = build_features_from_saldo(df_raw)

            # Misma lógica que en data_transformation: excluir fechas y targets
            excluded_cols = ["load_date", "target_saldo_tplus1", "target_diff_tplus1"]
            feature_cols = [c for c in df_feat_tmp.columns if c not in excluded_cols]

            # Guardamos para futuras ejecuciones
            os.makedirs(os.path.dirname(FEATURE_COLS_PATH), exist_ok=True)
            with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
                json.dump(feature_cols, f, ensure_ascii=False, indent=2)

            logging.info(
                "feature_cols reconstruido y guardado en %s (%d columnas).",
                FEATURE_COLS_PATH,
                len(feature_cols),
            )

        # ============================
        # 3) Cargar modelo CatBoost
        # ============================
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}.")

        model = joblib.load(MODEL_PATH)
        logging.info("Modelo CatBoost cargado desde %s", MODEL_PATH)

        # ============================
        # 4) Generar forecast
        # ============================
        df_forecast = forecast_next_n_days_catboost(
            model=model,
            df_history_raw=df_raw,
            feature_cols=feature_cols,
            n_days=n_days,
        )

        # ============================
        # 5) Guardar resultado
        # ============================
        os.makedirs(FORECAST_DIR, exist_ok=True)
        forecast_path = os.path.join(
            FORECAST_DIR, f"forecast_{n_days}d_catboost_saldo.csv"
        )

        df_forecast.to_csv(forecast_path, index=False)
        logging.info("Forecast guardado en %s", forecast_path)

        return forecast_path

    except Exception as e:
        # Envolvemos cualquier error en nuestro CustomException
        raise CustomException(e, sys) from e
