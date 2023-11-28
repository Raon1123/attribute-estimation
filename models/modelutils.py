import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.largeloss import LargeLossMatters, BoostCAM

import utils.logging as logging

def get_model(config, num_classes, use_feature=False):
    """
    Get the model from config

    Input
    - config: config variable for setting, see load_config() in main.py
    - num_classes: number of classes in the dataset
    """
    if config['LOGGING']['load_model']:
        model = logging.load_model(model, config)
        return model

    model_config = config['METHOD']
    model_name = model_config['name']

    if model_name == 'LargeLossMatters':
        try:
            model = LargeLossMatters(
                num_classes=num_classes,
                backbone=model_config['backbone'],
                freeze_backbone=model_config['freeze_backbone'],
                use_feature=use_feature,
                mod_schemes=model_config['mod_scheme'],
                delta_rel = model_config['delta_rel'],
            )
        except KeyError:
            raise KeyError("Please check your config file, requirements backbone, freeze_backbone, mod_schemes and delta_rel")
    elif model_name == 'BoostCAM':
        try:
            model = BoostCAM(
                num_classes=num_classes,
                backbone=model_config['backbone'],
                freeze_backbone=model_config['freeze_backbone'],
                use_feature=use_feature,
                alpha=model_config['alpha'],
            )
        except:
            raise KeyError("Please check your config file, requirements backbone, freeze_backbone, alpha")
    else:
        raise NotImplementedError
    
    return model


def get_optimizer(model, config):
    optim_config = config['OPTIMIZER']

    # load hyperparameter
    lr = optim_config['lr']
    try:
        momentum = optim_config['momentum']
    except KeyError:
        momentum = 0.9
    try:
        weight_decay = optim_config['weight_decay']
    except KeyError:
        weight_decay = 0.0005
    try:
        nestrov = optim_config['nestrov']
    except KeyError:
        nestrov = False

    if optim_config['name'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nestrov
        )
    elif optim_config['name'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    return optimizer


def get_scheduler(optimizer, config):
    try:
        scheduler_config = config['OPTIMIZER']['scheduler']
    except KeyError:
        return None
    
    if scheduler_config['name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_config['name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config['milestones'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_config['name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
    else:
        raise NotImplementedError

    return scheduler
