import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(preds, labels, config):
    """
    Get the loss function from the config.

    Args:
        preds (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Labels from the dataset.
        config (dict): Config dictionary.

    Returns:
        torch.Tensor: Loss value.
    """
    batch_size = preds.size(0)
    num_classes = preds.size(1)

    loss_matrix = F.binary_cross_entropy_with_logits(preds, labels.clip(0), reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(preds, torch.logical_not(labels.clip(0)).float(), reduction='none')

    
