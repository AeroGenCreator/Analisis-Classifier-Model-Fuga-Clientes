# Anlisis-Classifier-Model-Fuga-Clientes

🏦 Análisis de Fuga de Clientes en Beta Bank

![Image Alt](https://github.com/AeroGenCreator/Analisis-Classifier-Model-Fuga-Clientes/blob/main/1.png)

- Acceso al dashboard (Server gratuito, puede tardar en cargar la informacion la primera vez) [Dashboard](https://analisis-classifier-model-fuga-clientes.onrender.com)

💡 Descripción General del Proyecto

En Beta Bank, se identificó una preocupante tasa de fuga de clientes (abandono del banco). Dado que es más rentable retener a los clientes existentes que adquirir nuevos, el objetivo principal de este proyecto fue desarrollar un modelo de machine learning para predecir si un cliente dejará el banco en el futuro cercano.

El reto principal fue optimizar la calidad del modelo para alcanzar un valor de F1 score de al menos 0.59 en el conjunto de prueba.
🎯 Objetivos y Métricas Clave
Métrica	Descripción	Umbral Mínimo
F1 Score	Media armónica de la precisión y la sensibilidad. Mide el equilibrio entre falsos positivos y falsos negativos.	0.59
AUC-ROC	Área bajo la curva Característica Operativa del Receptor. Mide la capacidad del modelo para distinguir entre clases.	A comparar con F1
🛠️ Estructura del Proyecto y Metodología
1. Preparación y Exploración de Datos 📊

    Carga de Datos: Se cargó el conjunto de datos Churn.csv.

    Procesamiento:

        Eliminación de columnas no informativas (como RowNumber, CustomerId, Surname).

        Codificación One-Hot para variables categóricas (Geography, Gender).

    División: Los datos se dividieron en conjuntos de Entrenamiento, Validación y Prueba para garantizar una evaluación rigurosa.

    Análisis del Desequilibrio: Se examinó la distribución de la variable objetivo (Exited).

2. Entrenamiento Inicial del Modelo (Sin Corrección de Desequilibrio)

    Modelo Utilizado: Se entrenó un modelo de Bosque Aleatorio (Random Forest) o Árbol de Decisión (Decision Tree) inicial.

    Hallazgos: El modelo inicial, al no tener en cuenta el desequilibrio de clases (la minoría de clientes fugados), mostró un F1 Score bajo. Este resultado confirmó la necesidad de aplicar técnicas de corrección.

3. Mejora de la Calidad del Modelo (Corrección de Desequilibrio)

Para superar el bajo F1 Score inicial, se aplicaron al menos dos enfoques para corregir el desequilibrio de clases y se optimizaron los hiperparámetros:
Enfoques de Corrección de Desequilibrio Implementados:

    Ajuste del Hiperparámetro class_weight:

        Se utilizó el parámetro class_weight='balanced' en el modelo para asignar un peso mayor a los ejemplos de la clase minoritaria (clientes que se fueron).

    Sobre-muestreo (Oversampling) de la Clase Minoritaria:

        Se aplicaron técnicas para replicar sintéticamente ejemplos de la clase minoritaria en el conjunto de entrenamiento.

Proceso de Optimización:

    Se entrenaron y evaluaron diferentes modelos (por ejemplo, Regresión Logística, Árbol de Decisión, Bosque Aleatorio) en los conjuntos de Entrenamiento y Validación.

    Se realizó una búsqueda de hiperparámetros (por ejemplo, max_depth, n_estimators) para encontrar la configuración óptima que maximizara el F1 Score.

    Hallazgo del Mejor Modelo: El modelo de Bosque Aleatorio con el ajuste de class_weight y los hiperparámetros optimizados resultó ser el de mejor rendimiento en el conjunto de validación.

4. Prueba Final del Modelo 🧪

    El mejor modelo encontrado en la etapa de validación se probó en el conjunto de Prueba (datos nunca antes vistos).

    Se calcularon las métricas finales (F1 Score y AUC-ROC).

✅ Resultados Finales

El modelo final (un Bosque Aleatorio optimizado) superó exitosamente el umbral mínimo requerido en el conjunto de prueba.
Métrica	Valor Obtenido	Umbral Mínimo	Resultado
F1 Score	`0.594488188976378`	≥0.59	Éxito
AUC-ROC	`0.8546618030655062`

Por ejemplo, un valor de AUC-ROC cercano a 1 indica una excelente capacidad de discriminación, lo que complementa un F1 Score alto.
🚀 Conclusiones

Este proyecto demostró la importancia de:

    Tratar el desequilibrio de clases en problemas de clasificación para evitar un sesgo hacia la clase mayoritaria.

    Optimizar los hiperparámetros y seleccionar el modelo adecuado para el negocio.

El modelo resultante proporciona a Beta Bank una herramienta valiosa para identificar a los clientes en riesgo de fuga, permitiendo la implementación de estrategias proactivas de retención.
