import os
import pickle
import math

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from matplotlib import pyplot as plt

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
    logger = SummaryWriter(log_path)
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
  log_path = get_logger_path(config, subdir=exp_str(config))

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
  - ret: np.array
  """
  # img to np.array
  img = img.numpy().transpose(1, 2, 0).astype(np.uint8)
  heatmap = heatmap.numpy().astype(np.uint8)

  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  ret = cv2.addWeighted(heatmap, alpha, img, 1-alpha, 0)

  # img to pytorch tensor
  ret = TF.to_tensor(ret)
  return ret


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


def write_cams(config, imgs, cams, epoch, mode):
  """ 
  Write the file of cams
  """
  log_dir = get_logger_path(config, subdir=exp_str(config))

  img_file = f'{mode}img_{epoch}.pth'
  cam_file = f'{mode}cam_{epoch}.pth'

  img_path = os.path.join(log_dir, img_file)
  cam_path = os.path.join(log_dir, cam_file)

  torch.save(imgs, img_path)
  torch.save(cams, cam_path)


def log_cams(logger, imgs, cams, epoch, mode, config=None):
  """
  Logging the cams
  """

  for idx, (img, cam) in enumerate(zip(imgs, cams)):
    applied_imgs = []
    for attribute_cam in cam:
      applied_imgs.append(heatmap_on_image(img, attribute_cam, 0.7))

      # make grid
    img_grid = make_grid(applied_imgs, nrow=int(math.sqrt(imgs.shape[0])))

    img_name = f'{mode}{idx}_image'
    if logger == 'wandb':
      wandb.log({img_name: [wandb.Image(img_grid)]}, step=epoch)
    else:
      try:
        logger.add_image(img_name, img_grid, epoch)
      except:
        raise NotImplementedError
    
  # write image
  if config is not None:
    log_dir = get_logger_path(config, subdir=exp_str(config))
    img_file = f'{mode}img_{epoch}.png'
    img_path = os.path.join(log_dir, img_file)
    plt.savefig(grids, img_path)