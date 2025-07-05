import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import shap

# --- Cargar modelos y preprocesador ---
@st.cache_resource
def cargar_modelos():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
    with open("logistic_model.pkl", "rb") as f:
        logistic_model = cloudpickle.load(f)
    with open("random_forest_model.pkl", "rb") as f:
        rf_model = cloudpickle.load(f)
    with open("lightgbm_model.pkl", "rb") as f:
        lgb_model = cloudpickle.load(f)
    return {
        "Regresión Logística": logistic_model,
        "Random Forest": rf_model,
        "LightGBM": lgb_model,
    }, preprocessor

models, preprocessor = cargar_modelos()
feature_names = preprocessor.get_feature_names_out()

# --- Título ---
st.title("🔮 Predicción y Análisis de Churn en Fondos de Inversión")

# --- Sidebar: Inputs del usuario ---
st.sidebar.header("📊 Características del Cliente")

rendimiento = st.sidebar.slider("Rendimiento del fondo (%)", -10.0, 20.0, 5.0) / 100
volatilidad = st.sidebar.slider("Volatilidad", 0.01, 0.5, 0.2)
comisiones = st.sidebar.slider("Comisiones (%)", 0.1, 3.0, 1.0) / 100
benchmark = st.sidebar.slider("Comparación con benchmark (%)", -10.0, 10.0, 0.0) / 100
frecuencia = st.sidebar.slider("Frecuencia de transacciones por mes", 0, 10, 2)
tiempo = st.sidebar.slider("Tiempo de permanencia (años)", 0.0, 10.0, 2.0)
edad = st.sidebar.slider("Edad", 18, 80, 40)
patrimonio = st.sidebar.number_input("Patrimonio invertido", value=100000.0)
ubicacion = st.sidebar.selectbox("Ubicación", ["Capital", "Interior", "CABA", "Conurbano"])

modelo_elegido = st.selectbox("🧠 Elegí un modelo", list(models.keys()))

# --- Convertir input a DataFrame ---
input_df = pd.DataFrame([{
    "rendimiento": rendimiento,
    "volatilidad": volatilidad,
    "comisiones": comisiones,
    "comparacion_benchmark": benchmark,
    "frecuencia_transacciones": frecuencia,
    "tiempo_permanencia": tiempo,
    "edad": edad,
    "patrimonio": patrimonio,
    "ubicacion": ubicacion
}])

# --- Transformar input ---
X_input = preprocessor.transform(input_df)

# --- Tabs para Predicción e Interpretabilidad ---
tab_prediccion, tab_interpretabilidad = st.tabs(["🔮 Predicción", "📈 Interpretabilidad"])

with tab_prediccion:
    st.header("Predicción de Churn")
    modelo = models[modelo_elegido]
    prob = modelo.predict_proba(X_input)[0][1]
    st.markdown(f"### ✅ Probabilidad de churn: **{prob:.2%}**")

with tab_interpretabilidad:
    st.header("Interpretabilidad del Modelo")

    if modelo_elegido == "Regresión Logística":
        st.subheader("📑 Coeficientes, errores estándar y p-valores")

        df = pd.read_csv("datos_churn.csv")
        X_full = preprocessor.transform(df.drop("churn", axis=1))
        y_full = df["churn"].values

        def coeficientes_pvalores(log_model, X, y):
            X_design = np.hstack([np.ones((X.shape[0], 1)), X])
            p = log_model.predict_proba(X)[:, 1]
            V = np.diag(p * (1 - p))
            XtVX = X_design.T @ V @ X_design
            cov_matrix = np.linalg.inv(XtVX)

            coef = np.hstack([log_model.intercept_, log_model.coef_.flatten()])
            se = np.sqrt(np.diag(cov_matrix))
            z_scores = coef / se
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

            return pd.DataFrame({
                "Variable": ["Intercepto"] + list(feature_names),
                "Coeficiente": coef.round(4),
                "Error estándar": se.round(4),
                "p-valor": p_values.round(4)
            })

        coef_df = coeficientes_pvalores(models["Regresión Logística"], X_full, y_full)
        st.dataframe(coef_df)

    else:
        st.subheader("🌳 SHAP Values: Importancia de las variables")
        df = pd.read_csv("datos_churn.csv")
        X_full = preprocessor.transform(df.drop("churn", axis=1))

        try:
            # Probar TreeExplainer
            explainer = shap.TreeExplainer(models[modelo_elegido])
            shap_values = explainer.shap_values(X_full)

            # Detectar estructura de shap_values
            if isinstance(shap_values, list):
                shap_matrix = shap_values[1]
                expected_value = explainer.expected_value[1]
            else:
                shap_matrix = shap_values
                expected_value = explainer.expected_value

        except Exception:
            # Fallback a KernelExplainer si TreeExplainer falla
            st.warning("TreeExplainer no soportado, usando KernelExplainer (más lento).")
            explainer = shap.KernelExplainer(models[modelo_elegido].predict_proba, shap.sample(X_full, 100))
            shap_matrix = explainer.shap_values(X_full)[1]
            expected_value = explainer.expected_value[1]

        # Gráfico global summary
        st.write("#### SHAP Summary Plot (Global)")
        fig_global, ax = plt.subplots()
        shap.summary_plot(shap_matrix, X_full, feature_names=feature_names, show=False)
        st.pyplot(fig_global)

        # Gráfico local waterfall
        st.write("#### SHAP Waterfall Plot (Predicción actual)")
        shap_values_input = explainer.shap_values(X_input)
        if isinstance(shap_values_input, list):
            shap_row = shap_values_input[1][0]
            expected_row = explainer.expected_value[1]
        else:
            shap_row = shap_values_input[0]
            expected_row = explainer.expected_value

        fig_local, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(
            values=shap_row,
            base_values=expected_row,
            data=X_input[0],
            feature_names=feature_names
        ), show=False)
        st.pyplot(fig_local)

