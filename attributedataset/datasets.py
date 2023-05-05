import os
import pickle

import torch
from torch.utils.data import Dataset

from PIL import Image


class AttributeDataset(Dataset):
    def __init__(self,
                 img_root,
                 label_str,
                 img_file,
                 label,
                 transform=None) -> None:
        super().__init__()

        self.img_root = img_root
        self.label_str = label_str
        self.img_file = img_file
        self.label = label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_file)
    
    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_root, self.img_file[idx]) 
        image = Image.open(img_path).convert('RGB') # (H, W, C)
        label = self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_label_str(self):
        """
        Return the list of attribute names.
        """
        return self.label_str
    
    def get_num_classes(self):
        """
        Return the number of classes.
        """
        return len(self.label_str)
    

class FeatureDataset(Dataset):
    def __init__(self,
                 feature_root,
                 img_file,
                 label,
                 transform=None) -> None:
        super().__init__()

        self.feature_root = feature_root
        self.img_file = img_file
        self.label = label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_file)
    
    def __getitem__(self, idx: int):
        feature_path = os.path.join(self.feature_root, self.img_file[idx]) 
        feature = torch.load(feature_path) # (H, W, C)
        label = self.label[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label