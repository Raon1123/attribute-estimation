import pickle
import yaml

import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torchvision.models import resnet50

from attributedataset.datasets import AttributeDataset


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


def generate_feature(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # check if feature exists
    feature_path = config['DATASET']['feature_path']
    try:
        with open(feature_path, 'rb') as f:
            feature_dict = pickle.load(f)
        print('Feature exists! At:', feature_path)
        return
    except:
        pass

    print('Generating feature...', config['DATASET']['name'])

    train_dataset, test_dataset, num_classes = get_dataloader(config)

    model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-2]) # (N, 2048, 7, 7)
    model = model.to(device)

    model.eval()

    train_feature, test_feature = [], []
    train_label, test_label = [], []

    with torch.no_grad():
        for img, label in train_dataset:
            img = img.to(device)
            feature = model(img.unsqueeze(0)).squeeze(0)
            train_feature.append(feature.cpu())
            train_label.append(label)
        
        for img, label in test_dataset:
            img = img.to(device)
            feature = model(img.unsqueeze(0)).squeeze(0)
            test_feature.append(feature.cpu())
            test_label.append(label)

    train_feature = torch.stack(train_feature)
    test_feature = torch.stack(test_feature)

    train_label = torch.stack(train_label)
    test_label = torch.stack(test_label)

    # save feature
    feature_dict = {
        'label_str': train_dataset.label_str,
        'train_feature': train_feature,
        'test_feature': test_feature,
        'train_label': train_label,
        'test_label': test_label
    }

    with open(feature_path, 'wb') as f:
        pickle.dump(feature_dict, f)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    generate_feature(config)