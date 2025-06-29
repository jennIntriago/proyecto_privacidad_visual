# Proyecto de Clasificación de Privacidad Visual

## Resumen

Este proyecto aborda la clasificación automática de imágenes como "privadas" o "públicas" mediante la implementación y evaluación comparativa de tres arquitecturas de redes neuronales convolucionales (CNN). El objetivo es desarrollar un modelo robusto y preciso que pueda identificar contenido sensible en imágenes, una tarea fundamental en el campo de la privacidad y la seguridad de datos. Se exploran técnicas de transfer learning y fine-tuning para optimizar el rendimiento de los modelos.

# Modelos Implementados

ResNet50: Arquitectura residual profunda, utilizando una estrategia de fine-tuning en dos fases.

DenseNet121: Redes densamente conectadas que mejoran el flujo de gradientes, con fine-tuning en las capas superiores.

YOLOv8-cls: Versión de clasificación de la reconocida arquitectura YOLO, optimizada para un alto rendimiento.

## Dataset

Este proyecto utiliza el dataset VizWiz para la clasificación de privacidad. El dataset debe ser descargado y colocado en la estructura de directorios correcta antes de ejecutar el pre-procesamiento.

Fuente (URL): https://vizwiz.org/tasks-and-datasets/vizwiz-priv/

## Instalación

```bash
pip install -r requirements.txt


```
