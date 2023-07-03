import os

import numpy as np
from torch.utils.data import Dataset

from PIL import Image


class AttributeDataset(Dataset):
    def __init__(self,
                 img_root,
                 label_str,
                 img_file,
                 label,
                 masks=None,
                 transform=None) -> None:
        super().__init__()

        self.img_root = img_root
        self._label_str = label_str
        self.img_file = img_file
        self.label = label
        self.transform = transform
        self.masks = masks

    def __len__(self) -> int:
        return len(self.img_file)
    
    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_root, self.img_file[idx]) 
        image = Image.open(img_path).convert('RGB') # (H, W, C)
        label = self.label[idx]
        mask = None

        if self.masks is not None:
            mask = self.masks[idx]
            label = label * (1 - mask)

        if self.transform:
            image = self.transform(image)

        return image, label, mask
    
    @property
    def label_str(self):
        return self._label_str
    
    @property
    def num_classes(self):
        return len(self.label_str)
    
    def get_mask(self, idx):
        if self.masks is None:
            return None
        return self.masks[idx]
    
    def get_before_mask(self, idx):
        label = self.label[idx]
        return label
    
    def get_image_path(self, idx):
        return os.path.join(self.img_root, self.img_file[idx])
    

class FeatureDataset(Dataset):
    def __init__(self,
                 label_str,
                 feature,
                 label,
                 masks=None,
                 ) -> None:
        super().__init__()

        self._label_str = label_str
        self.feature = feature
        self.label = label
        self.masks = masks

    def __len__(self) -> int:
        return len(self.feature)
    
    def __getitem__(self, idx: int):
        feature = self.feature[idx]
        label = self.label[idx]
        mask = None

        if self.masks is not None:
            mask = self.masks[idx]
            label = label * (1 - mask)

        return feature, label, mask
    
    @property
    def label_str(self):
        return self._label_str
    
    @property
    def num_classes(self):
        return len(self.label_str)
    
    def get_mask(self, idx):
        if self.masks is None:
            return None
        return self.masks[idx]
    
    def get_before_mask(self, idx):
        label = self.label[idx]
        return label
    
    def get_image_path(self, idx):
        return os.path.join(self.img_root, self.img_file[idx])