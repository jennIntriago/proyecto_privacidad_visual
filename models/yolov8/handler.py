# -*- coding: utf-8 -*-
"""
Contiene funciones manejadoras para interactuar con la API de YOLOv8,
incluyendo la carga del modelo y el proceso de entrenamiento.
"""

from ultralytics import YOLO
from pathlib import Path

def load_yolo_model(model_path='yolov8n-cls.pt'):
    """
    Carga un modelo YOLOv8 desde una ruta específica.

    Args:
        model_path (str): Ruta al archivo del modelo (.pt) o un nombre de modelo
                          base como 'yolov8n-cls.pt' para empezar desde cero.

    Returns:
        YOLO: Un objeto del modelo YOLO listo para usar.
    """
    # Verifica si la ruta existe, si no, asume que es un modelo base
    if not Path(model_path).exists():
        print(f"Modelo base '{model_path}' seleccionado para entrenamiento.")
    else:
        print(f"Cargando modelo desde: {model_path}")
        
    model = YOLO(model_path)
    return model

def train_yolo_model(model, config):
    """
    Entrena un modelo YOLOv8 usando una configuración específica.

    Args:
        model (YOLO): El objeto del modelo YOLO a entrenar.
        config (dict): Un diccionario con todos los hiperparámetros de entrenamiento.

    Returns:
        Results: Un objeto con los resultados del entrenamiento.
    """
    print("🚀 Iniciando el entrenamiento del modelo YOLOv8...")
    
    results = model.train(
        # --- Parámetros de datos y entrenamiento ---
        data=config.get('data_path'),
        epochs=config.get('epochs', 100),
        patience=config.get('patience', 20),
        batch=config.get('batch_size', 16),
        imgsz=config.get('image_size', 224),
        
        # --- Configuración de ejecución ---
        device=config.get('device', 'cpu'),
        project=config.get('project_name', 'YOLOv8_Project'),
        name=config.get('run_name', 'exp'),
        exist_ok=True,
        
        # --- Hiperparámetros del optimizador ---
        optimizer=config.get('optimizer', 'AdamW'),
        lr0=config.get('learning_rate_initial', 0.001),
        lrf=config.get('learning_rate_final', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3.0),

        # --- Aumentación de datos ---
        augment=config.get('augment', True),
        hsv_h=config.get('aug_hsv_h', 0.015),
        hsv_s=config.get('aug_hsv_s', 0.7),
        hsv_v=config.get('aug_hsv_v', 0.4),
        degrees=config.get('aug_degrees', 0.5),
        translate=config.get('aug_translate', 0.1),
        scale=config.get('aug_scale', 0.5),
        shear=config.get('aug_shear', 0.5),
        flipud=config.get('aug_flipud', 0.0),
        fliplr=config.get('aug_fliplr', 0.5),
        mosaic=config.get('aug_mosaic', 1.0),
        mixup=config.get('aug_mixup', 0.0),

        # --- Opciones adicionales ---
        save=True,
        plots=True,
        verbose=True
    )
    
    print("✅ Entrenamiento completado.")
    return results