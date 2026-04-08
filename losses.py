import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Ожидает на вход нормализованные признаки (embeddings), а не логиты!
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # 1. L2-нормализация признаков (обязательный шаг для Contrastive Loss)
        features = F.normalize(features, p=2, dim=1)

        # 2. Матрица косинусного сходства
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 3. Маска для поиска позитивных пар (один и тот же класс)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 4. Удаляем диагональ (чтобы не сравнивать картинку саму с собой)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Численная стабильность (вычитаем максимум)
        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - sim_max.detach()

        # 5. Знаменатель (сумма по всем негативным и позитивным)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # 6. Среднее только по позитивным парам
        mask_sum = mask.sum(1)
        # Избегаем деления на ноль, если у картинки нет пары ее класса в батче
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 7. Финальный лосс
        loss = -mean_log_prob_pos
        return loss.mean()

def get_loss_function(name='focal', gamma=2.0, class_weights=None, device='cpu', temperature=0.07, label_smoothing=0.0):
    if class_weights is not None:
        # Убеждаемся, что веса находятся на том же устройстве, что и модель
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if name == 'ce':
        # Передаем label_smoothing в стандартную кросс-энтропию
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif name == 'focal':
        return FocalLoss(gamma=gamma, alpha=class_weights)
    elif name == 'supcon':
        # SupCon обычно не использует веса классов, он балансируется за счет батча
        return SupConLoss(temperature=temperature)
    else:
        raise ValueError("Поддерживаются только 'ce', 'focal' и 'supcon'")