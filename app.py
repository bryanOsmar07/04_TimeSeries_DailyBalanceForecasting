import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

from src.pipeline.forecast_pipeline import run_forecast_pipeline


# =========================
# Helpers
# =========================
def load_raw_data(raw_path: str = "artifacts/data_ingestion/raw.csv") -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos brutos en {raw_path}. "
            "Ejecuta primero el training_pipeline."
        )

    df = pd.read_csv(raw_path)
    if "load_date" not in df.columns or "saldo" not in df.columns:
        raise ValueError("raw.csv debe contener las columnas 'load_date' y 'saldo'.")
    df["load_date"] = pd.to_datetime(df["load_date"])
    df = df.sort_values("load_date").reset_index(drop=True)
    return df


def load_forecast(forecast_path: str) -> pd.DataFrame:
    df = pd.read_csv(forecast_path)
    if "load_date" not in df.columns or "saldo_pred" not in df.columns:
        raise ValueError(
            "El forecast debe contener las columnas 'load_date' y 'saldo_pred'."
        )
    df["load_date"] = pd.to_datetime(df["load_date"])
    df = df.sort_values("load_date").reset_index(drop=True)
    return df


def load_metrics(
    path: str = "artifacts/model_trainer/metrics_catboost_saldo.csv",
) -> pd.DataFrame:
    """
    Carga las métricas del modelo.

    Importante: usamos index_col=0 para que el índice quede como
    TRAIN / VAL / TEST (y no como 0,1,2).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el archivo de métricas en {path}. "
            "Ejecuta primero el training_pipeline."
        )
    df = pd.read_csv(path, index_col=0)
    return df


def load_feature_importances(
    path: str = "artifacts/model_trainer/feature_importances_catboost_saldo.csv",
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el archivo de importancias en {path}. "
            "Ejecuta primero el training_pipeline."
        )
    df = pd.read_csv(path)
    return df


def plot_hist_vs_forecast_plotly(df_raw: pd.DataFrame, df_forecast: pd.DataFrame, n_hist: int = 90):
    """
    Gráfico interactivo con Plotly.
    - Últimos n_hist días reales
    - Forecast futuro
    """

    # Últimos días reales
    hist_tail = df_raw.tail(n_hist).copy()

    fig = go.Figure()

    # ----------------------------
    # Línea: Saldo real
    # ----------------------------
    fig.add_trace(
        go.Scatter(
            x=hist_tail["load_date"],
            y=hist_tail["saldo"],
            mode="lines",
            name=f"Saldo real (últimos {n_hist} días)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Fecha: %{x}<br>Saldo: %{y:,.0f}<extra></extra>",
        )
    )

    # ----------------------------
    # Línea: Forecast
    # ----------------------------
    fig.add_trace(
        go.Scatter(
            x=df_forecast["load_date"],
            y=df_forecast["saldo_pred"],
            mode="lines",
            name=f"Forecast {len(df_forecast)} días",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="Fecha: %{x}<br>Predicción: %{y:,.0f}<extra></extra>",
        )
    )

    # ----------------------------
    # Línea vertical del último día real
    # ----------------------------
    last_real_date = hist_tail["load_date"].max()
    fig.add_vline(
        x=last_real_date,
        line_width=1.5,
        line_dash="dot",
        line_color="gray",
        annotation_text="Último día observado",
        annotation_position="top right",
    )

    # ----------------------------
    # Layout general
    # ----------------------------
    fig.update_layout(
        title="📈 Saldo diario – Histórico vs Pronóstico (Interactivo)",
        xaxis_title="Fecha",
        yaxis_title="Saldo",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
    )

    return fig


def plot_hist_vs_forecast(
    df_raw: pd.DataFrame, df_forecast: pd.DataFrame, n_hist: int = 90
):
    """
    Gráfico de últimos n_hist días reales + forecast.
    """
    hist_tail = df_raw.tail(n_hist).copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        hist_tail["load_date"],
        hist_tail["saldo"],
        label=f"Saldo real (últimos {n_hist} días)",
        linewidth=2,
    )

    ax.plot(
        df_forecast["load_date"],
        df_forecast["saldo_pred"],
        linestyle="--",
        linewidth=2,
        label=f"Forecast próximos {len(df_forecast)} días",
    )

    # Línea vertical en último día observado
    last_date = hist_tail["load_date"].max()
    ax.axvline(
        last_date,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Último día observado",
    )

    ax.set_title("Saldo diario – histórico vs pronóstico (CatBoost)", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Saldo")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_feature_importances(df_imp: pd.DataFrame, top_n: int = 20):
    """
    Barplot de top_n features por importancia.
    Se asume un CSV con columnas: ['feature', 'importance'].
    (Si tu archivo tiene otros nombres, ajusta aquí.)
    """
    # Intentar detectar nombres de columnas
    cols = [c.lower() for c in df_imp.columns]
    if "feature" in cols:
        feature_col = df_imp.columns[cols.index("feature")]
    else:
        feature_col = df_imp.columns[0]

    if "importance" in cols:
        imp_col = df_imp.columns[cols.index("importance")]
    elif "mean_abs_shap" in cols:
        imp_col = df_imp.columns[cols.index("mean_abs_shap")]
    else:
        imp_col = df_imp.columns[1]

    df_plot = df_imp[[feature_col, imp_col]].copy()
    df_plot = df_plot.sort_values(imp_col, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_plot[feature_col], df_plot[imp_col])
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} variables más importantes", fontsize=14)
    ax.set_xlabel(imp_col)
    fig.tight_layout()
    return fig


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(
        page_title="Forecast de Saldos Diarios",
        layout="wide",
    )

    st.title("📈 Forecast de Saldos Diarios – CatBoost")
    st.markdown(
        """
Este dashboard utiliza un modelo **CatBoost** entrenado sobre una serie temporal de saldos diarios
para pronosticar los próximos días, apoyado en:

- Features de calendario (día, mes, día de la semana, fin de mes, etc.)
- Flags bancarios (paydays, CTS, gratificación, utilidades, feriados)
- Lags y ventanas móviles del saldo
- Features adicionales de tendencia, volatilidad y shocks

---
"""
    )

    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    n_days = st.sidebar.slider(
        "Días a pronosticar", min_value=7, max_value=60, value=15, step=1
    )
    n_hist = st.sidebar.slider(
        "Días históricos a mostrar en el gráfico",
        min_value=30,
        max_value=180,
        value=90,
        step=10,
    )

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("🚀 Generar forecast")

    # Contenido principal
    tab1, tab2, tab3 = st.tabs(
        ["🔮 Forecast", "📊 Métricas del modelo", "📌 Importancia de variables"]
    )

    # =========================
    # TAB 1: Forecast
    # =========================
    with tab1:
        st.subheader("🔮 Pronóstico de saldos")

        try:
            df_raw = load_raw_data()
        except Exception as e:
            st.error(f"Error cargando raw.csv: {e}")
            return

        if run_btn:
            with st.spinner("Ejecutando forecast_pipeline..."):
                try:
                    forecast_path = run_forecast_pipeline(n_days=n_days)
                except Exception as e:
                    st.error(f"Error ejecutando el forecast_pipeline: {e}")
                    return

            st.success(f"Forecast generado correctamente: `{forecast_path}`")

            try:
                df_forecast = load_forecast(forecast_path)
            except Exception as e:
                st.error(f"Error cargando el forecast: {e}")
                return

            # Gráfico
            fig = plot_hist_vs_forecast(df_raw, df_forecast, n_hist=n_hist)
            st.pyplot(fig)
            #fig = plot_hist_vs_forecast_plotly(df_raw, df_forecast, n_hist=n_hist)
            #st.plotly_chart(fig, use_container_width=True)

            # Tabla
            st.markdown("### 📄 Detalle del forecast")
            st.dataframe(df_forecast)

        else:
            st.info(
                "Ajusta los parámetros en la barra lateral y pulsa "
                "**'🚀 Generar forecast'** para ejecutar el modelo."
            )

    # =========================
    # TAB 2: Métricas
    # =========================
    with tab2:
        st.subheader("📊 Métricas del modelo (CatBoost – saldo)")

        try:
            df_metrics = load_metrics()
        except Exception as e:
            st.error(f"No se pudieron cargar las métricas: {e}")
        else:
            st.markdown("Métricas calculadas en TRAIN / VAL / TEST:")
            st.dataframe(df_metrics)

            # Resumen rápido de Test
            if "split" in df_metrics.columns:
                mask_test = df_metrics["split"].str.upper() == "TEST"
                df_test = df_metrics[mask_test]
            else:
                # Usamos índice directamente: TRAIN / VAL / TEST
                if "TEST" in df_metrics.index:
                    df_test = df_metrics.loc[["TEST"]]
                else:
                    df_test = pd.DataFrame()

            if not df_test.empty:
                st.markdown("#### 🧪 Resumen Test")
                row = df_test.iloc[0]
                st.write(
                    f"**MAE:** {row['MAE']:,.0f}  |  "
                    f"**RMSE:** {row['RMSE']:,.0f}  |  "
                    f"**MAPE:** {row['MAPE']:.2f}%  |  "
                    f"**R²:** {row['R2']:.3f}"
                )

    # =========================
    # TAB 3: Importancia de variables
    # =========================
    with tab3:
        st.subheader("📌 Importancia de variables")

        try:
            df_imp = load_feature_importances()
        except Exception as e:
            st.error(f"No se pudieron cargar las importancias de variables: {e}")
        else:
            st.markdown("Top variables más relevantes según el modelo CatBoost:")

            fig_imp = plot_feature_importances(df_imp, top_n=20)
            st.pyplot(fig_imp)

            with st.expander("Ver tabla completa de importancias"):
                st.dataframe(df_imp)


if __name__ == "__main__":
    main()
