"""
Large Loss Matters in Weakly Supervised Multi-Label Classification (CVPR 2022) 
[Paper](https://arxiv.org/abs/2206.03740)
"""
import math
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class LargeLossMatters(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone='resnet50',
                 freeze_backbone=False,
                 mod_schemes='LL-R',
                 delta_rel=0.1):
      super().__init__()

      self.num_classes = num_classes
      self.mod_schemes = mod_schemes
      self.delta_rel = delta_rel / 100
      self.clean_rate = 1.0

      if backbone == 'resnet50':
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2]) # (N, 2048, 7, 7)
      else:
        raise NotImplementedError
      
      if freeze_backbone:
        for param in self.backbone.parameters():
          param.requires_grad = False
      else:
        for param in self.backbone.parameters():
          param.requires_grad = True
      
      self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
      features = self.backbone(x) # (N, 2048, 7, 7)

      # adaptive global average pooling
      features = F.adaptive_avg_pool2d(features, (1)).squeeze(-1).squeeze(-1) # (N, 2048, 1, 1)
      
      logits = self.fc(features) # (N, num_classes)
      return logits
    
    def loss(self, x, labels):
      """
      Args:
        x: (N, num_classes)
        labels: (N, num_classes) 
      """
      preds = self.forward(x)

      batch_size = int(x.shape[0])
      num_classes = int(x.shape[1])

      loss_fn = nn.BCEWithLogitsLoss(reduction='none')
      loss_matrix = loss_fn(preds, labels) # (N, num_classes)
      corrected_loss_matrix = loss_fn(preds, torch.logical_not(labels).float()) # (N, num_classes)
      correction_idx = None

      if self.clean_rate == 1.0:
        return loss_matrix.mean(), correction_idx

      if self.mod_schemes == 'LL-R':
        k = math.ceil(batch_size * num_classes * (1-self.clean_rate))
      elif self.mod_schemes == 'LL-Ct':
        k = math.ceil(batch_size * num_classes * (1-self.clean_rate))
      elif self.mod_schemes == 'LL-Cp':
        k = math.ceil(batch_size * num_classes * self.delta_rel)
      else:
        raise NotImplementedError
      
      unobserved_loss = (labels == 0).bool() * loss_matrix # (N)
      topk = torch.topk(unobserved_loss.flatten(), k)
      topk_lossval = topk.values[-1]
      correction_idx = torch.where(unobserved_loss > topk_lossval)

      if self.mod_schemes == 'LL-R':
        corrected_loss_matrix = torch.zeros_like(loss_matrix)

      loss_matrix = torch.where(unobserved_loss < topk_lossval, loss_matrix, corrected_loss_matrix)

      return loss_matrix.mean(), correction_idx
    
    def get_cam(self, x):
      features = self.backbone(x) # (N, 2048, 7, 7)
      CAM = F.conv2d(features, self.fc.weight.unsqueeze(-1).unsqueeze(-1)) # (N, num_classes, 1, 1)
      return CAM
    
    def unfreeze_backbone(self):
      for param in self.backbone.parameters():
          param.requires_grad = True

    def forward_linear(self, x):
      x = self.fc(x)
      return x
    
    def decrease_clean_rate(self):
      self.clean_rate = self.clean_rate - self.delta_rel