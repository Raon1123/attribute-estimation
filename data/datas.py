import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class AttributeDataset(Dataset):
    def __init__(self,
                 img_root,
                 label_str,
                 img_file,
                 label,
                 transforms=None) -> None:
        super().__init__()

        self.img_root = img_root
        self.label_str = label_str
        self.img_file = img_file
        self.label = label
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_file)
    
    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_root, self.img_file[idx])
        image = read_image(img_path)
        label = self.label[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, label
    
    def get_label_str(self):
        """
        Return the list of attribute names.
        """
        return self.label_str
    

def get_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        proc_dict = pickle.load(f)
    
    img_root = proc_dict['img_root']
    label_str = proc_dict['label_str']
    train_img_file = proc_dict['train_img_file']
    test_img_file = proc_dict['test_img_file']
    train_label = proc_dict['train_label']
    test_label = proc_dict['test_label']

    train_dataset = AttributeDataset(img_root, label_str, train_img_file, train_label)
    test_dataset = AttributeDataset(img_root, label_str, test_img_file, test_label)

    return train_dataset, test_dataset