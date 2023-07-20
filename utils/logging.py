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

def project_name(config):
  ret = [config['LOGGING']['project'], config['DATASET']['name']]
  ret = '_'.join(ret)
  return ret


def get_model_path(config):
  logging_config = config['LOGGING']
  model_path = logging_config['model_path']

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  subdir = project_name(config)
  model_path = os.path.join(model_path, subdir)
  if not os.path.exists(model_path):
    os.makedirs(model_path)

  return model_path


def get_logger_path(config, postfix=False):
  logging_config = config['LOGGING']
  logger_path = logging_config['log_dir']
  os.makedirs(logger_path, exist_ok=True)

  subdir = project_name(config)
  logger_path = os.path.join(logger_path, subdir)
  os.makedirs(logger_path, exist_ok=True)

  if postfix:
    logger_path = os.path.join(logger_path, logging_config['postfix'])
    os.makedirs(logger_path, exist_ok=True)

  return logger_path


def save_model(model, config):
  model_path = get_model_path(config)
  model_file = config['LOGGING']['postfix'] + '.pth'
  model_path = os.path.join(model_path, model_file)

  torch.save(model.state_dict(), model_path)


def load_model(model, config):
  model_path = get_model_path(config)
  model_file = config['LOGGING']['postfix'] + '.pth'
  model_path = os.path.join(model_path, model_file)

  model.load_state_dict(torch.load(model_path))

  return model


def logger_init(config):
  logging_config = config['LOGGING']
  log_dir = get_logger_path(config)

  postfix = logging_config['postfix']

  if wandb is not None and logging_config['logger'] == 'wandb':
    project = project_name(config)
    wandb.init(
        project=project,
        name=postfix,
        config=config
    )
    wandb.watch_called = False
    logger = 'wandb'
  elif logging_config['logger'] == 'tensorboard':
    logger = SummaryWriter(log_dir)
  else:
    raise NotImplementedError
  
  # config pickling
  pkl_file = f'{postfix}.pkl'
  pkl_path = os.path.join(log_dir, pkl_file)
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
  log_path = get_logger_path(config, postfix=True)

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
  - img: torch.tensor (C, H, W)
  - heatmap: torch.tensor (7, 7)
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
    

def print_metrics(metrics, prefix=''):
  print(prefix+'Metrics')
  for k, v in metrics.items():
    print(f'{prefix}{k}: {v.mean():.4f}')
  print()


def write_cams(config, imgs, cams, epoch, mode):
  """ 
  Write the file of cams
  """
  log_dir = get_logger_path(config, postfix=True)

  img_file = f'{mode}img_{epoch}.pth'
  cam_file = f'{mode}cam_{epoch}.pth'

  img_path = os.path.join(log_dir, img_file)
  cam_path = os.path.join(log_dir, cam_file)

  torch.save(imgs, img_path)
  torch.save(cams, cam_path)

  # interpolate cams as same size of image
  cams = torch.nn.functional.interpolate(cams, size=imgs.shape[2:], mode='bicubic')

  for idx, (img, cam) in enumerate(zip(imgs, cams)):
    applied_imgs = []
    num_class = cam.shape[0]
    for attribute_cam in cam:
      applied_imgs.append(heatmap_on_image(img, attribute_cam, 0.7))

    # make grid
    img_grid = make_grid(applied_imgs, nrow=int(math.sqrt(num_class)))
    img_grid = img_grid.permute(1, 2, 0).numpy()

    # save grid image
    grid_img_file = f'{mode}{idx}_{epoch}_grid.png'
    grid_img_path = os.path.join(log_dir, grid_img_file)
    plt.imshow(img_grid)
    plt.savefig(grid_img_path)


def log_cams(logger, imgs, cams, epoch, mode, config=None):
  """
  Logging the cams
  """
  grid_cam_imgs = []

  # interpolate cams as same size of image
  cams = torch.nn.functional.interpolate(cams, size=imgs.shape[2:], mode='bicubic')

  for idx, (img, cam) in enumerate(zip(imgs, cams)):
    applied_imgs = []
    num_class = cam.shape[0]
    for attribute_cam in cam:
      applied_imgs.append(heatmap_on_image(img, attribute_cam, 0.7))

      # make grid
    img_grid = make_grid(applied_imgs, nrow=int(math.sqrt(num_class)))
    grid_cam_imgs.append(img_grid)

    img_name = f'{mode}{idx}_{epoch}_image'
    if logger == 'wandb':
      wandb.log({img_name: [wandb.Image(img_grid)]}, step=epoch)
    else:
      try:
        logger.add_image(img_name, img_grid, epoch)
      except:
        raise NotImplementedError