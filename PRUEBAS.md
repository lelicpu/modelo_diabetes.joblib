# Pruebas del modelo y de la aplicación

Todas las pruebas automatizadas están en `test_app.py` y se ejecutan con `pytest -q test_app.py` (10 pruebas, todas aprobadas en la última ejecución).
Los valores de la columna «Resultado obtenido» salen de ejecutar la aplicación real (con `streamlit.testing`) y de las capturas de la carpeta `capturas/`.
Los registros de referencia (`casos_referencia.csv`) los genera la sección 14 del notebook con el mismo pipeline que carga la aplicación.

| Prueba | Entrada o procedimiento | Resultado esperado | Resultado obtenido |
|---|---|---|---|
| Entrada válida | Registro de referencia 1 (paciente diabética del conjunto de prueba): embarazos 8, glucosa 105, presión 100, pliegue 36, insulina «No disponible», IMC 43,3, pedigrí 0,239, edad 45 | La aplicación procesa los datos y presenta una salida comprensible, con el aviso académico visible | **Cumple.** Probabilidad 57,5 %, «perfil de riesgo alto», aviso académico visible. Evidencia: `capturas/01_entrada_valida.png` |
| Valores límite | Todos los campos en el mínimo permitido por la interfaz, y luego todos en el máximo | El sistema procesa y advierte cuando los valores exceden lo visto en entrenamiento | **Cumple.** Mínimos: 3,8 % (riesgo bajo) con advertencia de valores fuera del rango de entrenamiento. Máximos: 88,0 % (riesgo alto) con la misma advertencia. En ambos casos coincide con el cálculo directo del pipeline |
| Entrada incompleta | Las 5 mediciones opcionales marcadas «No disponible» | La aplicación no falla, imputa con la mediana del entrenamiento y orienta al usuario | **Cumple.** Probabilidad 47,1 % con el mensaje «Faltan 5 de las 5 mediciones opcionales: la predicción tiene poca fiabilidad». Evidencia: `capturas/02_entrada_incompleta.png` |
| Entrada inválida | Escribir una edad negativa (−5) | El valor inválido no llega al modelo | **Cumple.** El campo conserva un valor dentro de sus límites (29, la mediana) y la aplicación no falla |
| Valor fuera del dominio | Edad de 100 años (la interfaz la admite; el entrenamiento llegó a 81) | Procesa y advierte que la predicción es poco fiable | **Cumple.** Probabilidad 60,2 % con la advertencia «`age` = 100 (el modelo vio valores entre 21 y 81)». Evidencia: `capturas/03_valor_fuera_de_dominio.png` |
| Consistencia notebook – Streamlit | Los 3 registros de referencia ejecutados en el notebook y en la aplicación | La probabilidad y la clase deben coincidir | **Cumple.** Diferencia ≤ 1×10⁻⁹ en los tres: 0,57514 (clase 1), 0,28466 (clase 0) y 0,67761 (clase 1) |
| Acceso público | Abrir el enlace de Streamlit desde una sesión distinta (otro navegador o celular, sin sesión iniciada) | La aplicación carga y permite completar el flujo principal | **Pendiente.** Solo puede probarse después de desplegar la aplicación; completar esta fila con el resultado y una captura |

## Qué se probó, qué ocurrió y qué se corrigió

- Se probó que la interfaz envía al modelo exactamente las 8 columnas y el orden que espera el pipeline, y que las mediciones marcadas «No disponible» llegan como valores faltantes (NaN), que el pipeline imputa con la mediana del entrenamiento.
- Durante el desarrollo de las pruebas se supuso que un valor inválido provocaría un error; en realidad el campo numérico de Streamlit lo rechaza y conserva un valor dentro de sus límites, por lo que la prueba se ajustó a ese comportamiento.
- Al revisar las primeras capturas se vio que el script de captura había marcado la casilla equivocada («No disponible» de glucosa en lugar de insulina); se repitieron hasta que la captura 1 reprodujo el registro de referencia (57,5 %).
- Limitación conocida: las advertencias de fiabilidad son reglas simples (valores fuera del rango observado y número de mediciones ausentes); no cuantifican la incertidumbre de cada predicción individual.
