# -*- coding: utf-8 -*-
"""
Contiene toda la lógica para el pre-procesamiento de datos.

Este script realiza las siguientes acciones:
1. Carga los archivos de anotaciones (train, val, test).
2. Define las transformaciones de imágenes (data augmentation) para los modelos
   tipo clasificador (ResNet, DenseNet).
3. Define un `Dataset` personalizado de PyTorch para dichos modelos.
4. Define y ejecuta la función que reestructura los datos en el formato
   requerido por YOLOv8 para clasificación.
5. Se puede ejecutar como un script principal para preparar todos los datos.
"""

import os
import json
import shutil
import pandas as pd
from PIL import Image
from pathlib import Path

# --- CONFIGURACIÓN GLOBAL DE RUTAS ---
# Ajusta estas rutas según la estructura de tu proyecto.
# Se asume que este script se ejecuta desde la raíz del proyecto.
ANNOTATIONS_DIR = './data/raw/Annotations'
IMAGE_DIR = './data/raw/Filling_Images'
YOLO_OUTPUT_DIR = './data/processed/yolo_dataset'


def load_and_clean_dataframe(split):
    """
    Carga un archivo JSON de anotaciones, lo convierte a DataFrame y valida
    que las imágenes asociadas existan en el disco.

    Args:
        split (str): El nombre del conjunto de datos ('train', 'val', 'test').

    Returns:
        pd.DataFrame: Un DataFrame limpio con las rutas a las imágenes.
    """
    json_path = Path(ANNOTATIONS_DIR) / f"{split}.json"
    if not json_path.exists():
        print(f"Advertencia: No se encontró el archivo de anotaciones {json_path}")
        return None

    df = pd.read_json(json_path)
    
    # Crea la ruta completa a la imagen .png
    df['image_path'] = df['image'].apply(
        lambda x: Path(IMAGE_DIR) / (Path(x).stem + ".png")
    )
    
    # Filtra el dataframe para mantener solo las imágenes que existen
    original_count = len(df)
    df = df[df['image_path'].apply(lambda p: p.exists())].copy()
    cleaned_count = len(df)
    
    if original_count != cleaned_count:
        print(f"Para '{split}', se eliminaron {original_count - cleaned_count} "
              f"entradas por imágenes no encontradas.")
        
    print(f"DataFrame '{split}' cargado con {cleaned_count} muestras válidas.")
    return df


def preparar_dataset_yolo(annotations_dir, image_dir, output_dir):
    """
    Crea la estructura de directorios para YOLOv8 (train/val/test) y copia
    las imágenes en sus carpetas de clase correspondientes ('public'/'private').
    
    Esta función está adaptada de tu script original de YOLOv8.
    """
    print("\n🚀 Iniciando preparación del dataset para YOLOv8...")
    output_path = Path(output_dir)

    # Limpia el directorio de salida para un inicio limpio
    if output_path.exists():
        print(f"Limpiando directorio existente: {output_path}")
        shutil.rmtree(output_path)

    # Procesa cada split (train, val, test)
    for split in ['train', 'val', 'test']:
        json_path = Path(annotations_dir) / f'{split}.json'
        if not json_path.exists():
            print(f"⚠️ Archivo {json_path} no encontrado. Omitiendo split.")
            continue

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        print(f"📁 Procesando split: {split}...")
        count = 0
        for ann in annotations:
            # Construye el nombre de archivo .png
            image_filename_png = f'{Path(ann["image"]).stem}.png'
            is_private = ann['private']
            label = 'private' if is_private == 1 else 'public'

            # Define la carpeta de destino
            dst_folder = output_path / split / label
            dst_folder.mkdir(parents=True, exist_ok=True)

            # Define la ruta de origen y destino del archivo
            src_path = Path(image_dir) / image_filename_png
            dst_path = dst_folder / image_filename_png

            if src_path.exists():
                try:
                    # Copia el archivo
                    shutil.copy(src_path, dst_path)
                    count += 1
                except Exception as e:
                    print(f"   ❌ Error copiando imagen {src_path}: {e}")
            else:
                # Este caso ya se maneja en load_and_clean_dataframe, pero se
                # mantiene como una doble verificación.
                pass
        print(f"   ✅ Se copiaron {count} imágenes para el split '{split}'.")
                
    print("\n✅ Preparación del dataset YOLOv8 completada.")
    return str(output_path)


# --- Bloque principal para ejecución directa ---
if __name__ == '__main__':
    print("======================================================")
    print("=      INICIANDO SCRIPT DE PRE-PROCESAMIENTO DE DATOS      =")
    print("======================================================")
    
    # Paso 1: Verificar que los directorios de datos crudos existan
    if not Path(ANNOTATIONS_DIR).exists() or not Path(IMAGE_DIR).exists():
        print("\nERROR: Los directorios de datos crudos no existen.")
        print(f"Verifica que '{ANNOTATIONS_DIR}' y '{IMAGE_DIR}' sean correctos.")
    else:
        # Paso 2: Cargar y limpiar DataFrames (útil para ResNet/DenseNet)
        print("\n--- Cargando DataFrames de anotaciones ---")
        train_df = load_and_clean_dataframe('train')
        val_df = load_and_clean_dataframe('val')
        test_df = load_and_clean_dataframe('test')
        
   

        # Paso 3: Preparar el dataset con la estructura de carpetas para YOLOv8
        preparar_dataset_yolo(ANNOTATIONS_DIR, IMAGE_DIR, YOLO_OUTPUT_DIR)

    print("\n======================================================")
    print("=          PRE-PROCESAMIENTO FINALIZADO          =")
    print("======================================================")