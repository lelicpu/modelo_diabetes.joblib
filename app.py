"""Aplicación de Streamlit – Predicción académica de diabetes (Proyecto de aula, Minería de Datos, CUC).

La aplicación NO entrena ni transforma datos por su cuenta: carga `modelo_diabetes.joblib`, que contiene
el mismo `Pipeline` (imputación + modelo) evaluado en `Proyecto_Diabetes_Final.ipynb`, y le envía las
mismas ocho columnas, en el mismo orden, que espera el pipeline.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RUTA_MODELO = Path(__file__).parent / "modelo_diabetes.joblib"

# clave, etiqueta, mínimo, máximo, paso, formato, ¿admite "No disponible"?, ayuda
CAMPOS = [
    ("pregnancies", "Número de embarazos", 0, 25, 1, "%d", False, "Embarazos previos de la paciente."),
    ("glucose", "Glucosa (mg/dL)", 20, 300, 1, "%d", True,
     "Concentración de glucosa en plasma a las 2 horas de una prueba de tolerancia oral."),
    ("bloodpressure", "Presión arterial diastólica (mm Hg)", 20, 150, 1, "%d", True, "Presión diastólica."),
    ("skinthickness", "Pliegue cutáneo del tríceps (mm)", 1, 100, 1, "%d", True, "Espesor del pliegue cutáneo."),
    ("insulin", "Insulina sérica a las 2 h (µU/ml)", 1, 900, 1, "%d", True, "Insulina a las 2 horas."),
    ("bmi", "Índice de masa corporal (kg/m²)", 10.0, 70.0, 0.1, "%.1f", True, "Peso (kg) / estatura (m)²."),
    ("diabetes_pedigree_function", "Función de pedigrí de diabetes", 0.0, 2.5, 0.001, "%.3f", False,
     "Puntaje que resume los antecedentes familiares de diabetes."),
    ("age", "Edad (años)", 1, 110, 1, "%d", False, "El modelo se entrenó con mujeres de 21 años o más."),
]

st.set_page_config(page_title="Predicción académica de diabetes", page_icon="🩺", layout="centered")


@st.cache_resource
def cargar_artefacto():
    return joblib.load(RUTA_MODELO)


def valor_inicial(clave, minimo, maximo, medianas):
    mediana = medianas[clave]
    if isinstance(minimo, int):
        return int(min(max(round(mediana), minimo), maximo))
    return float(min(max(round(mediana, 3), minimo), maximo))


st.title("🩺 Predicción académica de diabetes")
st.warning(
    "**Uso académico.** Esta aplicación es un ejercicio del curso de Minería de Datos (Universidad de la Costa). "
    "Sus resultados **no constituyen un diagnóstico y no sustituyen una valoración profesional de salud**."
)

if not RUTA_MODELO.exists():
    st.error("No se encontró el archivo `modelo_diabetes.joblib`. Ejecute primero el notebook "
             "`Proyecto_Diabetes_Final.ipynb` para generarlo.")
    st.stop()

artefacto = cargar_artefacto()
pipeline = artefacto["pipeline"]
umbral = artefacto["umbral"]
variables = artefacto["variables"]
rangos = artefacto["rangos_entrenamiento"]
medianas = artefacto["medianas_entrenamiento"]
met = artefacto["metricas_prueba"]
ic = artefacto["ic95_prueba"]

with st.sidebar:
    st.header("Sobre el modelo")
    st.write(f"**Modelo:** {artefacto['nombre_modelo']}")
    st.write(f"**Umbral de decisión:** {umbral:.2f}")
    st.write("**Datos:** Pima Indians Diabetes (768 mujeres de 21 años o más, ascendencia Pima).")
    st.markdown("**Desempeño en prueba (154 registros):**")
    st.write(f"- Recall (diabéticas): {met['Recall (1)']:.3f} (IC 95 %: {ic['Recall (1)'][0]:.2f}–{ic['Recall (1)'][1]:.2f})")
    st.write(f"- Precision: {met['Precision (1)']:.3f}")
    st.write(f"- F1-score: {met['F1-score (1)']:.3f}")
    st.write(f"- AUC-ROC: {met['AUC-ROC']:.3f}")
    st.caption("Los resultados no son generalizables a otras poblaciones (hombres, menores de edad, otras etnias).")

st.subheader("Datos de la paciente")
st.caption("Si un dato no se midió, marque «No disponible»: el modelo lo reemplaza con la mediana del entrenamiento, "
           "igual que durante la evaluación.")

valores = {}
no_disponibles = []
columnas = st.columns(2)
for i, (clave, etiqueta, minimo, maximo, paso, formato, opcional, ayuda) in enumerate(CAMPOS):
    with columnas[i % 2]:
        marcado_na = bool(opcional and st.session_state.get(f"na_{clave}", False))
        valor = st.number_input(etiqueta, min_value=minimo, max_value=maximo,
                                value=valor_inicial(clave, minimo, maximo, medianas),
                                step=paso, format=formato, help=ayuda, key=clave, disabled=marcado_na)
        if opcional:
            st.checkbox("No disponible", key=f"na_{clave}")
    if opcional and st.session_state.get(f"na_{clave}", False):
        valores[clave] = np.nan
        no_disponibles.append(clave)
    else:
        valores[clave] = float(valor)

if st.button("Evaluar perfil", type="primary"):
    # Se arma una fila con EXACTAMENTE las columnas y el orden que espera el pipeline
    entrada = pd.DataFrame([[valores[v] for v in variables]], columns=variables)
    proba = float(pipeline.predict_proba(entrada)[0, 1])
    clase = int(proba >= umbral)
    st.session_state["ultima_probabilidad"] = proba
    st.session_state["ultima_clase"] = clase

    st.subheader("Resultado del modelo")
    st.metric("Probabilidad estimada de diabetes", f"{proba:.1%}")
    if clase == 1:
        st.error(f"Clasificación del modelo: **perfil de riesgo alto** (probabilidad ≥ {umbral:.2f}). "
                 "Se sugeriría una prueba confirmatoria con un profesional de salud.")
    else:
        st.success(f"Clasificación del modelo: **perfil de riesgo bajo** (probabilidad < {umbral:.2f}). "
                   "Esto no descarta la enfermedad.")

    # Advertencias sobre la calidad de la entrada
    fuera = []
    for clave, (minimo, maximo) in rangos.items():
        v = valores[clave]
        if not np.isnan(v) and (v < minimo or v > maximo):
            fuera.append(f"`{clave}` = {v:g} (el modelo vio valores entre {minimo:g} y {maximo:g})")
    if fuera:
        st.warning("**Valores fuera del rango de entrenamiento:** " + "; ".join(fuera) +
                   ". La predicción es poco fiable para estos valores.")
    if len(no_disponibles) >= 3:
        st.warning(f"Faltan {len(no_disponibles)} de las 5 mediciones opcionales: la predicción tiene **poca fiabilidad**.")
    elif "glucose" in no_disponibles:
        st.warning("Sin glucosa, la variable más influyente, la predicción es poco fiable.")

    with st.expander("Valores enviados al modelo"):
        mostrado = entrada.T.rename(columns={0: "Valor"})
        mostrado["Valor"] = [("No disponible (imputado)" if np.isnan(x) else f"{x:g}") for x in entrada.iloc[0]]
        st.dataframe(mostrado)
