import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.largeloss import LargeLossMatters

import utils.logging as logging

def get_model(config, num_classes):
    if config['LOGGING']['load_model']:
        model = logging.load_model(model, config)
        return model

    model_config = config['METHOD']

    if model_config['name'] == 'LargeLossMatters':
        try:
            model = LargeLossMatters(
                num_classes=num_classes,
                backbone=model_config['backbone'],
                freeze_backbone=model_config['freeze_backbone'],
                mod_schemes=model_config['mod_scheme'],
                delta_rel = model_config['delta_rel'],
            )
        except KeyError:
            raise KeyError("Please check your config file, requirements backbone, freeze_backbone, mod_schemes and delta_rel")
    else:
        raise NotImplementedError
    
    return model


def get_optimizer(model, config):
    optim_config = config['OPTIMIZER']

    if optim_config['name'] == 'SGD':
        try:
            nestrov = optim_config['nestrov']
        except KeyError:
            nestrov = False
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=optim_config['lr'],
            momentum=optim_config['momentum'],
            weight_decay=optim_config['weight_decay'],
            nesterov=nestrov
        )
    elif optim_config['name'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optim_config['lr'],
            weight_decay=optim_config['weight_decay']
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

    return scheduler
