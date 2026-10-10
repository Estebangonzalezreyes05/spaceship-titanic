# Spaceship Titanic — Sistemas Inteligentes

Proyecto de clasificación binaria que predice la variable `Transported` a partir de los datos de pasajeros de Spaceship Titanic. Incluye limpieza de faltantes, ingeniería de variables, entrenamiento y evaluación.

## Contenido

- `convert_pdf.py`: reconstruye CSV a partir de las tablas paginadas de `train.pdf`, `test.pdf` y `sample_submission.pdf`.
- `train_model.py`: entrena regresión logística y Random Forest; selecciona por F1 de validación, entrena con todos los datos etiquetados y crea las predicciones.
- `data/*.csv`: datos reconstruidos desde los PDF proporcionados.
- `outputs/submission.csv`: predicciones de prueba.
- `outputs/metricas.json`: resultados de validación.
- `outputs/matriz_confusion.png`: evaluación visual.

## Requisitos e instalación en macOS

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

Si los CSV incluidos ya están disponibles:

```bash
python train_model.py
```

Para reconstruirlos de nuevo desde los PDF, copia los tres archivos en la raíz del proyecto y ejecuta:

```bash
python convert_pdf.py --pdf-dir . --out-dir data
python train_model.py
```

## Advertencia de integridad de datos

Los PDF entregados contienen **8,505** filas de entrenamiento y **4,185** de prueba. La versión habitual del desafío Kaggle tiene **8,693** y **4,277** respectivamente. Por tanto, las predicciones generadas desde estos PDF **no constituyen una entrega completa para el concurso original**. Para competir en Kaggle, utiliza `train.csv` y `test.csv` originales de Kaggle y vuelve a ejecutar `train_model.py`; asegúrate de que `sample_submission.csv` también sea el original.

Los PDF presentan también texto de nombres truncado por maquetación; la columna `Name` **no se usa en el modelo**. Los archivos de origen deben tratarse como documentos proporcionados para la práctica, no como copias completas certificadas de Kaggle.

## Evitar fuga de información

Se separa una validación estratificada 80/20. Las imputaciones, escalado y One-Hot Encoding se ajustan **solo en el conjunto de entrenamiento** mediante un `Pipeline`. La evaluación se efectúa antes del ajuste final con todas las filas etiquetadas.

## Subir a GitHub

```bash
git init
git add .
git commit -m "Proyecto ML Spaceship Titanic"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/spaceship-titanic.git
git push -u origin main
```

Antes de `git remote add origin`, crea un repositorio vacío con ese nombre desde tu cuenta de GitHub. Sustituye `TU_USUARIO` por tu nombre de usuario. Para la autenticación, usa GitHub CLI (`gh auth login`) o un método de acceso autorizado; GitHub no acepta la contraseña normal de la cuenta para operaciones Git HTTPS.
