"""Pruebas automatizadas de la aplicación de Streamlit y de su consistencia con el notebook.

Ejecutar desde la carpeta del repositorio:   pytest -q test_app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).parent
ARTEFACTO = joblib.load(RAIZ / "modelo_diabetes.joblib")
PIPELINE, UMBRAL, VARIABLES = ARTEFACTO["pipeline"], ARTEFACTO["umbral"], ARTEFACTO["variables"]
CASOS = pd.read_csv(RAIZ / "casos_referencia.csv")
OPCIONALES = ["glucose", "bloodpressure", "skinthickness", "insulin", "bmi"]


def ejecutar_app(valores, no_disponibles=()):
    """Abre la aplicación, rellena los campos y pulsa «Evaluar perfil»."""
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    assert not at.exception
    for clave in no_disponibles:
        at.checkbox(key=f"na_{clave}").check()
    at.run()
    for clave, valor in valores.items():
        if clave not in no_disponibles:
            at.number_input(key=clave).set_value(valor)
    at.run()
    at.button[0].click().run()
    return at


def proba_directa(valores):
    """Probabilidad calculada directamente con el pipeline (lo que haría el notebook)."""
    fila = pd.DataFrame([[valores.get(v, np.nan) for v in VARIABLES]], columns=VARIABLES)
    return float(PIPELINE.predict_proba(fila)[0, 1])


def valores_de_caso(fila):
    return {v: (None if pd.isna(fila[v]) else float(fila[v])) for v in VARIABLES}


def test_la_app_usa_el_pipeline_evaluado():
    assert list(PIPELINE.feature_names_in_) == VARIABLES
    assert ARTEFACTO["version_scikit_learn"]


@pytest.mark.parametrize("indice", range(len(CASOS)))
def test_consistencia_notebook_streamlit(indice):
    """El mismo registro debe dar la misma probabilidad y clase en el notebook y en Streamlit."""
    fila = CASOS.iloc[indice]
    valores = valores_de_caso(fila)
    faltantes = [k for k, v in valores.items() if v is None]
    at = ejecutar_app({k: v for k, v in valores.items() if v is not None}, faltantes)
    assert not at.exception
    assert at.session_state["ultima_probabilidad"] == pytest.approx(fila["probabilidad_notebook"], abs=1e-9)
    assert at.session_state["ultima_clase"] == int(fila["clase_notebook"])


def test_entrada_valida_muestra_resultado():
    fila = CASOS.iloc[0]
    at = ejecutar_app({k: v for k, v in valores_de_caso(fila).items() if v is not None},
                      [k for k, v in valores_de_caso(fila).items() if v is None])
    assert not at.exception
    assert len(at.metric) == 1 and "%" in at.metric[0].value
    assert len(at.error) + len(at.success) == 1          # una clasificación comprensible
    assert any("no sustituyen" in w.value for w in at.warning)   # aviso académico visible


@pytest.mark.parametrize("extremo", ["min", "max"])
def test_valores_limite(extremo):
    """Valores en los límites que permite la interfaz: procesa y advierte, sin fallar."""
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    valores = {}
    for campo in at.number_input:
        valores[campo.key] = campo.min if extremo == "min" else campo.max
        campo.set_value(valores[campo.key])
    at.run()
    at.button[0].click().run()
    assert not at.exception
    p = at.session_state["ultima_probabilidad"]
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(proba_directa(valores), abs=1e-9)
    assert any("fuera del rango" in w.value for w in at.warning)   # los límites exceden lo visto en entrenamiento


def test_entrada_incompleta_no_falla_y_advierte():
    """Omitir las 5 mediciones opcionales: la app no falla, imputa y avisa de la poca fiabilidad."""
    base = valores_de_caso(CASOS.iloc[0])
    base = {k: (0.0 if v is None else v) for k, v in base.items()}
    at = ejecutar_app({k: base[k] for k in VARIABLES if k not in OPCIONALES}, OPCIONALES)
    assert not at.exception
    esperado = proba_directa({k: base[k] for k in VARIABLES if k not in OPCIONALES})
    assert at.session_state["ultima_probabilidad"] == pytest.approx(esperado, abs=1e-9)
    assert any("poca fiabilidad" in w.value for w in at.warning)


def test_valor_fuera_del_dominio_de_entrenamiento():
    """Edad de 100 años (la interfaz la admite, el entrenamiento llegó a 81): procesa y advierte."""
    valores = valores_de_caso(CASOS.iloc[1])
    valores = {k: v for k, v in valores.items() if v is not None}
    valores["age"] = 100
    at = ejecutar_app(valores, [k for k in OPCIONALES if k not in valores])
    assert not at.exception
    assert any("fuera del rango" in w.value and "age" in w.value for w in at.warning)


def test_la_interfaz_no_acepta_valores_por_debajo_del_minimo():
    """Un valor inválido (edad = -5) no llega al modelo: el campo conserva un valor dentro de sus límites."""
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    at.number_input(key="age").set_value(-5).run()
    assert not at.exception
    assert at.number_input(key="age").value >= at.number_input(key="age").min
    at.button[0].click().run()
    assert not at.exception
    assert 0.0 <= at.session_state["ultima_probabilidad"] <= 1.0
