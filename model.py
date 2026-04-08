import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class JointSkinLesionClassifier(nn.Module):
    """
    Модель с двумя головами для совместного обучения (Classification + SupCon).
    Возвращает как логиты для CE/Focal Loss, так и эмбеддинги для Contrastive Loss.
    """
    def __init__(self, num_classes=7, projection_dim=128):
        super().__init__()
        
        # 1. ОБЩИЙ БЭКБОН (ConvNeXt-Large)
        self.backbone = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        
        # Удаляем стандартный слой классификации ConvNeXt
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity() 
        
        # 2. ГОЛОВА КЛАССИФИКАЦИИ (Для CrossEntropy / Focal Loss)
        self.classification_head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.GELU(), 
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

        # 3. ГОЛОВА ПРОЕКЦИИ (Для Supervised Contrastive Loss)
        # Сжимает фичи в компактное 128D пространство
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x, return_embeddings=True):
        # Пропускаем через бэкбон
        features = self.backbone(x)
        
        # Получаем предсказания классов
        logits = self.classification_head(features)
        
        # Во время валидации/инференса нам нужны только логиты
        if not return_embeddings:
            return logits
            
        # Во время обучения (Joint Training) получаем эмбеддинги
        embeddings = self.projection_head(features)
        # Обязательная L2-нормализация для косинусного расстояния в SupCon
        embeddings = F.normalize(embeddings, dim=1)
        
        return logits, embeddings

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Убеждаемся, что обе головы разморожены
        for param in self.classification_head.parameters():
            param.requires_grad = True
        for param in self.projection_head.parameters():
            param.requires_grad = True
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True