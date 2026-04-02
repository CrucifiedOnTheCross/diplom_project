import torch.nn as nn
from torchvision import models

class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Переключаемся на ConvNeXt-Large
        self.backbone = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        
        # У ConvNeXt слой классификации - это Sequential, где Linear имеет индекс 2
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity() 
        
        # Heavy Head (остается прежней)
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.GELU(), 
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True