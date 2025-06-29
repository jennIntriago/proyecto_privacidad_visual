# -*- coding: utf-8 -*-
"""
Contiene la definición de la arquitectura del modelo DenseNet121 para
la clasificación de privacidad de imágenes.
"""

import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNetPrivacy(nn.Module):
    """
    Define la arquitectura DenseNet121 para clasificación binaria.
    
    Carga un modelo pre-entrenado y realiza un fine-tuning selectivo
    descongelando solo las últimas capas del extractor de características.
    """
    def __init__(self, num_classes, dropout_rate):
        """
        Inicializa el modelo.

        Args:
            num_classes (int): Número de clases de salida.
            dropout_rate (float): Tasa de dropout para el clasificador.
        """
        super().__init__()

        # Carga el modelo DenseNet121 pre-entrenado en ImageNet
        self.net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Descongela únicamente las capas de los bloques 3 y 4 para fine-tuning
        for name, param in self.net.named_parameters():
            is_trainable = "features.denseblock3" in name or "features.denseblock4" in name
            param.requires_grad = is_trainable

        # Reemplaza la cabeza del clasificador por una nueva
        in_features = self.net.classifier.in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """Define el pase hacia adelante del modelo."""
        return self.net(x)