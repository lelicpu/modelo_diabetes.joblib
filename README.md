# Predicción académica de diabetes – Proyecto de aula de Minería de Datos (CUC)

Clasificación binaria de diabetes con el *Pima Indians Diabetes Dataset* (768 registros, 8 variables clínicas). Entrega final (corte 3): notebook completo, modelo exportado y aplicación en Streamlit.

> **Uso académico.** Los resultados no constituyen un diagnóstico ni sustituyen una valoración profesional de salud.

- **Aplicación de Streamlit:** `<pegar aquí el enlace público después de desplegar>`
- **Repositorio:** `<pegar aquí el enlace de GitHub>`

## Contenido

| Archivo | Descripción |
|---|---|
| `Proyecto_Diabetes_Final.ipynb` | Notebook final ejecutado: datos, limpieza, EDA, línea base, ajuste de hiperparámetros, umbral, evaluación, análisis de errores, exportación del modelo, discusión y declaración de uso de IA. |
| `diabetes_dataset.csv` | Dataset (768 × 9). |
| `modelo_diabetes.joblib` | Pipeline entrenado y evaluado en el notebook (imputación + modelo), umbral y metadatos. Lo carga la aplicación. |
| `app.py` | Aplicación de Streamlit. |
| `test_app.py` | Pruebas automatizadas de la aplicación y de la consistencia notebook–Streamlit. |
| `casos_referencia.csv` | Registros de referencia con la salida del notebook. |
| `PRUEBAS.md` | Tabla de pruebas con resultados reales. |
| `capturas/` | Capturas de la interfaz. |
| `figs/` | Figuras generadas por el notebook. |
| `requirements.txt` / `requirements-notebook.txt` | Dependencias de la aplicación / del notebook completo. |

## Ejecución local

```bash
pip install -r requirements-notebook.txt
jupyter nbconvert --to notebook --execute --inplace Proyecto_Diabetes_Final.ipynb   # regenera modelo y figuras
pytest -q test_app.py                                                              # pruebas
streamlit run app.py                                                               # aplicación
```

`scikit-learn` está fijado en la versión con la que se guardó `modelo_diabetes.joblib` (1.9.1). Si se usa otra versión, ejecutar primero el notebook para regenerar el modelo.

## Resultados principales (conjunto de prueba, 154 registros)

Modelo final: Random Forest balanceado y ajustado (selección por F1 en validación cruzada), umbral 0,50. Recall de la clase diabética 0,759 · precision 0,594 · F1 0,667 · accuracy 0,734 · AUC-ROC 0,820. Los intervalos de confianza y la comparación con la línea base y con los modelos del avance 2 están en las secciones 9 a 13 del notebook.

## Despliegue en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio público de GitHub (incluyendo `modelo_diabetes.joblib`).
2. En <https://share.streamlit.io> crear una app nueva: repositorio, rama `main`, archivo principal `app.py`.
3. En *Advanced settings* elegir Python 3.12 o 3.13 y verificar que use `requirements.txt`.
4. Abrir el enlace desde una sesión distinta y completar la fila «Acceso público» de `PRUEBAS.md`.
