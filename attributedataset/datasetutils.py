import pickle
import yaml

import numpy as np
import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torchvision.models import resnet50

from attributedataset.datasets import AttributeDataset, FeatureDataset
from models.modelutils import get_model


def get_transforms(config):
    """
    Get the transforms for training and testing.

    Input
    - config: config variable for setting, see load_config() in main.py
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    dataset_name = config['DATASET']['name']
    input_size = config['DATASET']['transforms']['input_size']
    input_ratio = config['DATASET']['transforms']['input_ratio']
    resize = (int(input_size*input_ratio), input_size)

    train_transform = transforms.Compose([
        transforms.Resize(resize, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(resize, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return train_transform, test_transform


def get_dataset(config):
    """
    Get the dataset for training and testing.

    Input
    - config: config variable for setting, see load_config() in main.py
    """
    try:
        use_feature = config['DATASET']['use_feature']
    except:
        use_feature = False

    # use features?
    if use_feature:
        pkl_path = config['DATASET']['feature_path']
    else:
        pkl_path = config['DATASET']['pkl_path']

    try:
        with open(pkl_path, 'rb') as f:
            proc_dict = pickle.load(f)
    except:
        pkl_path = config['DATASET']['pkl_path']
        use_feature = False
        with open(pkl_path, 'rb') as f:
            proc_dict = pickle.load(f)

    if not use_feature:
        img_root = proc_dict['img_root']
        train_img_file = proc_dict['train_img_file']
        test_img_file = proc_dict['test_img_file']
    else:
        train_feature = proc_dict['train_feature']
        test_feature = proc_dict['test_feature']

    label_str = proc_dict['label_str']
    train_label = proc_dict['train_label']
    train_mask = proc_dict['train_mask']
    test_label = proc_dict['test_label']

    num_classes = len(label_str)

    if not use_feature:
        train_transform, test_transform = get_transforms(config)
        train_dataset = AttributeDataset(img_root, 
                                        label_str, 
                                        train_img_file, 
                                        train_label,
                                        train_mask,
                                        transform=train_transform)
        train_unmask_dataset = AttributeDataset(img_root,
                                                label_str,
                                                train_img_file,
                                                train_label,
                                                masks=None,
                                                transform=train_transform)
        test_dataset = AttributeDataset(img_root, 
                                        label_str, 
                                        test_img_file,
                                        test_label,
                                        transform=test_transform)
    else:
        train_dataset = FeatureDataset(label_str,
                                       train_feature,
                                       train_label,
                                       train_mask)
        train_unmask_dataset = FeatureDataset(label_str,
                                              train_feature,
                                              train_label,
                                              masks=None)
        test_dataset = FeatureDataset(label_str,
                                      test_feature,
                                      test_label)

    return train_dataset, train_unmask_dataset, test_dataset, num_classes


def get_dataloader(config):
    """
    Get the dataloader for training and testing.

    Input
    - config: config variable for setting, see load_config() in main.py
    """
    train_dataset, train_unmask_dataset, test_dataset, num_classes = get_dataset(config)

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
    """
    Generate the feature vector for training and testing.
    Make sure the feature_path is defined in config file.
    
    Input
    - config: config variable for setting, see load_config() in main.py
    """
    device = config['device']

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

    train_dataloader, test_dataloader, num_classes = get_dataloader(config)

    model = get_model(config, num_classes, use_feature=False).to(device)

    train_feature, test_feature = [], []
    train_label, test_label = [], []

    with torch.no_grad():
        for img, label in train_dataloader:
            img = img.to(device)
            feature = model.feature(img)
            train_feature.append(feature.cpu())
            train_label.append(label)
        
        for img, label in test_dataloader:
            img = img.to(device)
            feature = model.feature(img)
            test_feature.append(feature.cpu())
            test_label.append(label)

    train_feature = torch.cat(train_feature, dim=0)
    test_feature = torch.cat(test_feature, dim=0)

    train_label = torch.cat(train_label, dim=0).numpy().astype(np.float16)
    test_label = torch.cat(test_label, dim=0).numpy().astype(np.float16)

    assert train_feature.shape[0] == train_label.shape[0]
    assert test_feature.shape[0] == test_label.shape[0]
    assert train_label.shape[1] == test_label.shape[1] == num_classes

    # save feature
    feature_dict = {
        'label_str': train_dataloader.dataset.label_str,
        'train_feature': train_feature,
        'test_feature': test_feature,
        'train_label': train_label,
        'test_label': test_label
    }

    with open(feature_path, 'wb') as f:
        pickle.dump(feature_dict, f)

    print('Feature saved! At:', feature_path)
    print('Train feature shape:', train_feature.shape)
    print('Test feature shape:', test_feature.shape)
    print('Train label shape:', train_label.shape)
    print('Test label shape:', test_label.shape)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    generate_feature(config)