import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
    

def get_transforms(config):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    dataset_name = config['DATASET']['name']

    if dataset_name == 'rap1':
        train_transform = transforms.Compose([
            transforms.Resize((448, 224), antialias=None),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((448, 224), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((448,448), antialias=None),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((448,448), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    
    return train_transform, test_transform


def get_dataset(config):
    pkl_path = config['DATASET']['pkl_path']

    with open(pkl_path, 'rb') as f:
        proc_dict = pickle.load(f)
    
    img_root = proc_dict['img_root']
    label_str = proc_dict['label_str']
    train_img_file = proc_dict['train_img_file']
    test_img_file = proc_dict['test_img_file']
    train_label = proc_dict['train_label']
    test_label = proc_dict['test_label']

    num_classes = len(label_str)

    train_transform, test_transform = get_transforms(config)

    train_dataset = AttributeDataset(img_root, 
                                     label_str, 
                                     train_img_file, 
                                     train_label,
                                     transform=train_transform)
    test_dataset = AttributeDataset(img_root, 
                                    label_str, 
                                    test_img_file,
                                    test_label,
                                    transform=test_transform)

    return train_dataset, test_dataset, num_classes


def get_dataloader(config):
    train_dataset, test_dataset, num_classes = get_dataset(config)

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

