import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


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
        image = read_image(img_path).float() # (C, H, W)
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
    

def get_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        proc_dict = pickle.load(f)
    
    img_root = proc_dict['img_root']
    label_str = proc_dict['label_str']
    train_img_file = proc_dict['train_img_file']
    test_img_file = proc_dict['test_img_file']
    train_label = proc_dict['train_label']
    test_label = proc_dict['test_label']

    num_classes = len(label_str)

    transform = transforms.Compose([
        transforms.Resize((256, 128), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AttributeDataset(img_root, 
                                     label_str, 
                                     train_img_file, 
                                     train_label,
                                     transform=transform)
    test_dataset = AttributeDataset(img_root, 
                                    label_str, 
                                    test_img_file,
                                    test_label,
                                    transform=transform)

    return train_dataset, test_dataset, num_classes


def get_dataloader(config):
    train_dataset, test_dataset, num_classes = get_dataset(config['DATASET']['pkl_path'])

    loader_config = config['loader']

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=loader_config['batch_size'],
        shuffle=True,
        num_workers=loader_config['num_workers']
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=loader_config['batch_size'],
        shuffle=False,
        num_workers=loader_config['num_workers']
    )

    return train_dataloader, test_dataloader, num_classes

