# 🧠 Detector Multilabel IA con Aprendizaje Activo

Esta es una aplicación de Inteligencia Artificial capaz de detectar múltiples objetos en una misma imagen (Aviones, Autos, Barcos).
Incluye un sistema de **Active Learning** que permite re-entrenar el modelo desde la propia interfaz web.

## Características
- 🕵️‍♂️ **Multilabel:** Detecta varios objetos a la vez (ej: Barco + Avión).
- 📊 **Confianza:** Muestra barras de probabilidad en tiempo real.
- 🎓 **Re-entrenamiento:** Interfaz para corregir a la IA y mejorar el modelo automáticamente.
- 🎨 **Interfaz Moderna:** HTML/CSS limpio y responsivo.

## Estructura
- `app/`: Código fuente de la API (FastAPI) y Templates.
- `notebooks/`: Experimentos y entrenamiento inicial (Jupyter).
- `data/`: Dataset de imágenes.

## Instalación
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar: `cd app && uvicorn main:app --reload`