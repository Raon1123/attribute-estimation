import os
import pickle
import math

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from torch.utils.tensorboard import Summarylogger
try:
  import wandb
except ImportError:
  wandb = None

def exp_str(config):
  logging_config = config['LOGGING']
  log_str = [logging_config['project'], logging_config['postfix']]
  log_str = '_'.join(log_str)
  
  return log_str


def get_model_path(config):
  logging_config = config['LOGGING']
  model_path = logging_config['model_path']

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  return model_path


def get_logger_path(config, subdir=''):
  logging_config = config['LOGGING']
  logger_path = logging_config['log_dir']

  if not os.path.exists(logger_path):
    os.makedirs(logger_path)

  if subdir != '':
    logger_path = os.path.join(logger_path, subdir)
    if not os.path.exists(logger_path):
      os.makedirs(logger_path)

  return logger_path


def save_model(model, config):
  model_path = get_model_path(config)
  model_file = exp_str(config) + '.pth'
  model_path = os.path.join(model_path, model_file)

  torch.save(model.state_dict(), model_path)


def load_model(model, config):
  model_path = get_model_path(config)
  model_file = exp_str(config) + '.pth'
  model_path = os.path.join(model_path, model_file)

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
    logger = 'wandb'
  elif logging_config['logger'] == 'tensorboard':
    log_path = os.path.join(logging_config['log_dir'], log_str)
    logger = Summarylogger(log_path)
  else:
    raise NotImplementedError
  
  # config pickling
  pkl_file = f'{log_str}_config.pkl'
  pkl_path = os.path.join(logging_config['log_dir'], pkl_file)
  with open(pkl_path, 'wb') as f:
    pickle.dump(config, f)
  
  return logger


def log_loss(logger, loss, epoch, mode, config=None):
  if logger == 'wandb':
    wandb.log({f'{mode}_loss': loss}, step=epoch)
  else:
    try:
      logger.add_scalar(f'{mode}_loss', loss, epoch)
    except:
      raise NotImplementedError
  

def log_metrics(logger, metrics, epoch, config=None):
  """
  logging metrics
  Input
  - logger: tensorboard or wandb
  - metrics: dict
  - epoch: int
  """
  # metrics value mean
  for k, v in metrics.items():
    metrics[k] = v.mean()

  if logger == 'wandb':
    wandb.log(metrics, step=epoch)
  else:
    try:
      for k, v in metrics.items():
        logger.add_scalar(k, v, epoch)
    except:
      raise NotImplementedError
    

def write_metrics(gt, preds, metrics, epoch, config):
  """
  write metrics, gt, preds
  Input
  - gt: np.array
  - preds: np.array
  - metrics: dict
  """
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


def heatmap_on_image(img, heatmap, alpha=0.5):
  """
  draw heatmap on image
  Input
  - img: torch.tensor
  - heatmap: torch.tensor
  - alpha: float
  Output
  - img: np.array
  """
  
  # heatmap: H, W
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)

  # img to pytorch tensor
  img = TF.to_tensor(img)
  return img


def log_image(logger, images, epoch, mode, config=None):
  """
  logging images, log pytorch tensor and save torch tensor
  Input
  - logger: tensorboard or wandb
  - images: np.array
  - epoch: int
  """
  # save image torch
  save_dir = get_logger_path(config, subdir=exp_str(config))
  save_file = f'{mode}_image_{epoch}.pth'
  save_path = os.path.join(save_dir, save_file)

  torch.save(images, save_path)

  for sample in images:
    # sample: num_classes, C, H, W
    # make grid
    grid = make_grid(sample, nrow=int(math.sqrt(sample.shape[0])))

    if logger == 'wandb':
      wandb.log({f'{mode}_image': [wandb.Image(grid)]}, step=epoch)
    else:
      try:
        logger.add_image(f'{mode}_image', grid, epoch)
      except:
        raise NotImplementedError
    

def print_metrics(metrics, prefix=''):
  print(prefix+'Metrics')
  for k, v in metrics.items():
    print(f'{prefix}{k}: {v.mean():.4f}')
  print()


def write_cams():
  """ 
  Write the file of cams
  """
  pass


def log_cams():
  """
  Logging the cams
  """