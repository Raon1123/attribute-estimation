import os
import pickle

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
try:
  import wandb
except ImportError:
  wandb = None

def exp_str(config):
  logging_config = config['LOGGING']
  log_str = [logging_config['project'], logging_config['postfix']]
  log_str = '_'.join(log_str)
  
  return log_str


def save_model(model, config):
  logging_config = config['LOGGING']
  log_str = exp_str(config)

  save_dir = logging_config['model_path']
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  model_path = os.path.join(save_dir, log_str)

  torch.save(model.state_dict(), model_path)


def load_model(model, config):
  logging_config = config['LOGGING']
  log_str = exp_str(config)
  model_path = os.path.join(logging_config['model_path'], log_str)

  model.load_state_dict(torch.load(model_path))

  return model


def logger_init(config):
  logging_config = config['LOGGING']
  log_str = exp_str(config)

  if wandb is not None and logging_config['logger'] == 'wandb':
    project = logging_config['project'] + '_' + config['DATASET']['name']
    wandb.init(
        project=project,
        name=logging_config['postfix'],
        config=config
    )
    wandb.watch_called = False
    writer = 'wandb'
  elif logging_config['logger'] == 'tensorboard':
    log_path = os.path.join(logging_config['log_dir'], log_str)
    writer = SummaryWriter(log_path)
  else:
    raise NotImplementedError
  
  # config pickling
  pkl_file = f'{log_str}_config.pkl'
  pkl_path = os.path.join(logging_config['log_dir'], pkl_file)
  with open(pkl_path, 'wb') as f:
    pickle.dump(config, f)
  
  return writer


def log_loss(writer, loss, epoch, mode, config=None):
  if writer == 'wandb':
    wandb.log({f'{mode}_loss': loss}, step=epoch)
  else:
    try:
      writer.add_scalar(f'{mode}_loss', loss, epoch)
    except:
      raise NotImplementedError
  

def log_metrics(writer, metrics, epoch, config=None):
  # metrics value mean
  for k, v in metrics.items():
    metrics[k] = v.mean()

  if writer == 'wandb':
    wandb.log(metrics, step=epoch)
  else:
    try:
      for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)
    except:
      raise NotImplementedError
    

def write_metrics(gt, preds, metrics, epoch, config):
  logging_config = config['LOGGING']
  log_str = exp_str(config)
  log_path = os.path.join(logging_config['log_dir'], log_str)

  # make directory
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  gt_path = os.path.join(log_path, f'gt_{epoch}.pth')
  preds_path = os.path.join(log_path, f'preds_{epoch}.pth')

  np.save(gt_path, gt)
  np.save(preds_path, preds)

  metrics_path = os.path.join(log_path, f'metrics_{epoch}.pkl')
  with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)