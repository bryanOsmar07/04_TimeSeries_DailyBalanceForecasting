# src/components/data_transformation.py

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    """
    Configuración para la etapa de transformación:
    - Lee raw.csv generado por DataIngestion
    - Genera features y splits train/val/test
    """

    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")

    transformed_dir: str = os.path.join("artifacts", "data_transformation")

    train_output_path: str = os.path.join(transformed_dir, "train_transformed.csv")
    val_output_path: str = os.path.join(transformed_dir, "val_transformed.csv")
    test_output_path: str = os.path.join(transformed_dir, "test_transformed.csv")

    feature_cols_path: str = os.path.join(transformed_dir, "feature_cols.json")


def build_features_from_saldo(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Replica (lo más fiel posible) el pipeline del notebook:
      1) create_features_no_leakage
      2) df_diff con diff / target_diff_tplus1
      3) df_tuning con features adicionales (holidays, shock, etc.)
    Devuelve un df tipo df_tuning:
      - load_date
      - saldo
      - features
      - target_saldo_tplus1
      - target_diff_tplus1
    """

    import holidays

    # =========================
    # 0. Base
    # =========================
    df = df_raw.copy()
    if "load_date" not in df.columns or "saldo" not in df.columns:
        raise ValueError("El DataFrame debe contener 'load_date' y 'saldo'.")

    df["load_date"] = pd.to_datetime(df["load_date"])
    df = df.sort_values("load_date").reset_index(drop=True)

    # --------------------------------
    # 1) BLOQUE: create_features_no_leakage
    # --------------------------------
    df_cf = df.copy()
    df_cf = df_cf.set_index("load_date")

    # Calendario
    df_cf["year"] = df_cf.index.year
    df_cf["month"] = df_cf.index.month
    df_cf["day_of_month"] = df_cf.index.day
    df_cf["day_of_year"] = df_cf.index.dayofyear
    df_cf["dow"] = df_cf.index.dayofweek

    df_cf["is_weekend"] = (df_cf["dow"] >= 5).astype(int)
    df_cf["is_month_start"] = df_cf.index.is_month_start.astype(int)
    df_cf["is_month_end"] = df_cf.index.is_month_end.astype(int)

    # Paydays (versión notebook)
    df_cf["is_payday"] = df_cf["day_of_month"].isin([15, 30, 31]).astype(int)
    df_cf["is_pre_payday"] = df_cf["day_of_month"].isin([14, 29, 30]).astype(int)
    df_cf["is_post_payday"] = df_cf["day_of_month"].isin([1, 16]).astype(int)

    # Bonos por mes (sin día)
    df_cf["is_cts"] = df_cf["month"].isin([5, 11]).astype(int)
    df_cf["is_grati"] = df_cf["month"].isin([7, 12]).astype(int)
    df_cf["is_utilidades"] = df_cf["month"].isin([3, 4]).astype(int)

    # Feriados fijos (versión notebook)
    df_cf["is_christmas"] = (
        (df_cf["month"] == 12) & (df_cf["day_of_month"] == 25)
    ).astype(int)
    df_cf["is_fiestas_patrias"] = (
        (df_cf["month"] == 7) & (df_cf["day_of_month"] == 28)
    ).astype(int)

    # Lags saldo
    for lag in [1, 2, 3, 7, 14, 21]:
        df_cf[f"saldo_lag{lag}"] = df_cf["saldo"].shift(lag)

    # Rolling SIN leakage (usando saldo.shift(1))
    for w in [3, 7, 14, 30]:
        df_cf[f"rolling_mean_{w}"] = df_cf["saldo"].shift(1).rolling(w).mean()
        df_cf[f"rolling_std_{w}"] = df_cf["saldo"].shift(1).rolling(w).std()

    # Dif y pct_change SIN leakage
    df_cf["diff_1"] = df_cf["saldo"].diff(1).shift(1)
    df_cf["pct_change_1"] = df_cf["saldo"].pct_change(1).shift(1)
    df_cf["diff_7"] = df_cf["saldo"].diff(7).shift(1)

    # Target saldo t+1
    df_cf["target_saldo_tplus1"] = df_cf["saldo"].shift(-1)

    df_clean = df_cf.dropna().reset_index()  # load_date vuelve como columna

    # --------------------------------
    # 2) BLOQUE: df_diff (differences)
    # --------------------------------
    df_diff = df_clean.copy()

    # diff y target_diff_tplus1 (como en notebook)
    df_diff["diff"] = df_diff["saldo"].diff()
    df_diff["target_diff_tplus1"] = df_diff["diff"].shift(-1)
    df_diff = df_diff.dropna().copy()

    # Lags de diff
    for lag in [1, 2, 3, 7, 14, 21]:
        df_diff[f"diff_lag{lag}"] = df_diff["diff"].shift(lag)

    # Rolling de diff (sin shift en notebook)
    df_diff["diff_roll_mean_3"] = df_diff["diff"].rolling(3).mean()
    df_diff["diff_roll_std_3"] = df_diff["diff"].rolling(3).std()
    df_diff["diff_roll_mean_7"] = df_diff["diff"].rolling(7).mean()
    df_diff["diff_roll_std_7"] = df_diff["diff"].rolling(7).std()
    df_diff["diff_roll_mean_14"] = df_diff["diff"].rolling(14).mean()
    df_diff["diff_roll_std_14"] = df_diff["diff"].rolling(14).std()

    df_diff = df_diff.dropna().copy()

    # Esto es equivalente a tu df_model
    df_model = df_diff.copy()

    # --------------------------------
    # 3) BLOQUE: df_tuning (features extra)
    # --------------------------------
    df_tuning = df_model.copy()

    # 3.1 Feriados Perú (holidays package)
    pe_holidays = holidays.Peru(years=[2022, 2023, 2024, 2025])
    pe_holidays_dates = pd.to_datetime(list(pe_holidays.keys()))
    df_tuning["is_holiday"] = df_tuning["load_date"].isin(pe_holidays_dates).astype(int)

    df_tuning["is_business_day"] = (
        (df_tuning["dow"] < 5) & (df_tuning["is_holiday"] == 0)
    ).astype(int)

    df_tuning["is_zero_expected"] = (df_tuning["is_business_day"] == 0).astype(int)
    df_tuning["lag_is_zero_expected"] = df_tuning["is_zero_expected"].shift(1)

    # 3.2 Ciclo mensual (versión notebook)
    df_tuning["month_progress"] = (
        df_tuning["day_of_month"] / df_tuning["day_of_month"].max()
    )
    df_tuning["is_first_week"] = (df_tuning["day_of_month"] <= 7).astype(int)
    df_tuning["is_last_week"] = (df_tuning["day_of_month"] >= 24).astype(int)

    # 3.3 Dif suavizada
    df_tuning["diff_smooth_3"] = df_tuning["diff"].rolling(3).mean().shift(1)
    df_tuning["diff_smooth_7"] = df_tuning["diff"].rolling(7).mean().shift(1)

    q1 = df_tuning["diff"].quantile(0.01)
    q99 = df_tuning["diff"].quantile(0.99)
    df_tuning["diff_winsor"] = df_tuning["diff"].clip(q1, q99)

    # 3.4 Tendencias
    df_tuning["saldo_trend_7"] = df_tuning["saldo"].diff(7).shift(1)
    df_tuning["saldo_trend_30"] = df_tuning["saldo"].diff(30).shift(1)

    df_tuning["saldo_ema_7"] = df_tuning["saldo"].ewm(span=7).mean().shift(1)
    df_tuning["saldo_ema_30"] = df_tuning["saldo"].ewm(span=30).mean().shift(1)

    # 3.5 Shocks / volatilidad
    roll_std_30 = df_tuning["diff"].rolling(30).std()

    df_tuning["is_shock"] = (
        (df_tuning["diff"].abs() > roll_std_30 * 2).shift(1).fillna(0).astype(int)
    )

    df_tuning["cum_7"] = df_tuning["diff"].rolling(7).sum().shift(1)
    df_tuning["cum_14"] = df_tuning["diff"].rolling(14).sum().shift(1)
    df_tuning["cum_30"] = df_tuning["diff"].rolling(30).sum().shift(1)

    df_tuning["diff_abs"] = df_tuning["diff"].abs()

    df_tuning["repeat_shock"] = (
        (df_tuning["diff_abs"].shift(1) > roll_std_30).fillna(0).astype(int)
    )

    # 3.6 One-hot de día de semana
    for d in range(7):
        df_tuning[f"is_dow_{d}"] = (df_tuning["dow"] == d).astype(int)

    # Limpieza final
    df_tuning = df_tuning.dropna().reset_index(drop=True)

    return df_tuning


class DataTransformation:
    def __init__(self, config: Optional[DataTransformationConfig] = None) -> None:
        """
        Si no se pasa config, usa la configuración por defecto.
        Esto permite que los tests usen rutas temporales sin pisar artifacts reales.
        """
        self.config = config or DataTransformationConfig()

    def initiate_data_transformation(self):
        """
        Lee raw.csv, genera features, hace split temporal 70/15/15
        y devuelve X/y para train, val y test + lista de features.
        """
        logging.info("Iniciando Data Transformation...")

        try:
            raw_path = self.config.raw_data_path

            if not os.path.exists(raw_path):
                msg = f"No se encontró el archivo raw: {raw_path}"
                logging.error(msg)
                raise FileNotFoundError(msg)

            logging.info(f"Leyendo raw data desde: {raw_path}")
            df_raw = pd.read_csv(raw_path)

            # =========================
            # 1. Generar features
            # =========================
            logging.info("Construyendo features a partir de saldo...")
            df_feat = build_features_from_saldo(df_raw)

            logging.info(f"Shape con features: {df_feat.shape}")

            # =========================
            # 2. Definir target y features
            # =========================
            target_col = "target_saldo_tplus1"

            excluded_cols = [
                "load_date",
                "target_saldo_tplus1",
                "target_diff_tplus1",
            ]

            feature_cols = [c for c in df_feat.columns if c not in excluded_cols]

            X = df_feat[feature_cols]
            y = df_feat[target_col]

            # =========================
            # 3. Split temporal 70 / 15 / 15
            # =========================
            n = len(df_feat)
            train_size = int(n * 0.70)
            val_size = int(n * 0.15)

            X_train = X.iloc[:train_size].copy()
            y_train = y.iloc[:train_size].copy()

            X_val = X.iloc[train_size : train_size + val_size].copy()
            y_val = y.iloc[train_size : train_size + val_size].copy()

            X_test = X.iloc[train_size + val_size :].copy()
            y_test = y.iloc[train_size + val_size :].copy()

            logging.info(
                f"Splits -> "
                f"X_train: {X_train.shape}, "
                f"X_val: {X_val.shape}, "
                f"X_test: {X_test.shape}"
            )

            # =========================
            # 4. Guardar datasets transformados
            # =========================
            os.makedirs(self.config.transformed_dir, exist_ok=True)

            # Guardamos junto con load_date y el target
            df_train_out = df_feat.iloc[:train_size].copy()
            df_val_out = df_feat.iloc[train_size : train_size + val_size].copy()
            df_test_out = df_feat.iloc[train_size + val_size :].copy()

            df_train_out.to_csv(self.config.train_output_path, index=False)
            df_val_out.to_csv(self.config.val_output_path, index=False)
            df_test_out.to_csv(self.config.test_output_path, index=False)

            logging.info("Datasets transformados guardados")

            # Guardamos lista de features para reuso
            with open(self.config.feature_cols_path, "w", encoding="utf-8") as f:
                json.dump(feature_cols, f, ensure_ascii=False, indent=2)

            logging.info("feature_cols guardado correctamente.")

            # =========================
            # 5. Devolver arrays y metadata
            # =========================
            return (
                X_train.values,
                y_train.values,
                X_val.values,
                y_val.values,
                X_test.values,
                y_test.values,
                feature_cols,
            )

        except Exception as e:
            raise CustomException(e, sys)
