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

  pkl_file = config['DATASET']['pkl_file']
  pkl_file = pkl_file.split('_')
  split_seed = pkl_file[2]
  split_seed = f'seed{split_seed}'
  postfix = [split_seed, logging_config['postfix']]
  postfix = '_'.join(postfix)

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


def log_loss(logger, loss_dict, epoch, config=None):
  if logger == 'wandb':
    wandb.log(loss_dict, step=epoch)
  else:
    try:
      logger.add_scalar(loss_dict, epoch)
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

  gt_path = os.path.join(log_path, f'gt_{epoch}')
  preds_path = os.path.join(log_path, f'preds_{epoch}')

  np.save(gt_path, gt)
  np.save(preds_path, preds)

  metrics_path = os.path.join(log_path, f'metrics_{epoch}.pkl')
  with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)


def unnormalize(img):
    """
    unnormalize image
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    mn = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    st = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img * st + mn
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    return img


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
  img = unnormalize(img).astype(np.uint8)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  heatmap = heatmap.numpy().astype(np.uint8)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

  ret = cv2.addWeighted(heatmap, alpha, img, 1-alpha, 0)
  ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)

  # img to pytorch tensor
  ret = TF.to_tensor(ret)
  return ret
    

def print_metrics(metrics, prefix=''):
  print(prefix+'Metrics')
  for k, v in metrics.items():
    print(f'{prefix}{k}: {v.mean():.4f}')
  print()


def label_list_to_str(labels, label_to_str, threshold=0.5):
    """
    Boolean label list to string list.
    """
    ret = []
    labels = (labels > threshold)
    for i, label in enumerate(labels):
        if label == 1:
          ret.append(label_to_str[i])
    ret = ', '.join(ret)
    return ret 


def write_cams(config, 
               img_list, cam_list, target_list, pred_list, mask_list,
               epoch, mode,
               label_str):
  """ 
  Write the file of cams
  """
  log_dir = get_logger_path(config, postfix=True)

  # write pth file
  img_file = f'{mode}img_{epoch}.pth'
  cam_file = f'{mode}cam_{epoch}.pth'
  target_file = f'{mode}target_{epoch}.pth'
  pred_file = f'{mode}pred_{epoch}.pth'
  mask_file = f'{mode}mask_{epoch}.pth'

  img_path = os.path.join(log_dir, img_file)
  cam_path = os.path.join(log_dir, cam_file)
  target_path = os.path.join(log_dir, target_file)
  pred_path = os.path.join(log_dir, pred_file)
  mask_path = os.path.join(log_dir, mask_file)

  torch.save(img_list, img_path)
  torch.save(cam_list, cam_path)
  torch.save(target_list, target_path)
  torch.save(pred_list, pred_path)
  torch.save(mask_list, mask_path)

  # interpolate cams as same size of image
  cam_list = torch.nn.functional.interpolate(cam_list, size=img_list.shape[2:], mode='bicubic')

  # write grid image
  num_classes = len(label_str)
  num_rows = int(math.sqrt(num_classes)) + 1
  num_cols = int(math.sqrt(num_classes)) + 1
  
  num_figs = img_list.shape[0]

  for fig_idx in range(num_figs):
    fig = plt.figure(figsize=(20, 20))

    img = img_list[fig_idx]
    cam = cam_list[fig_idx]

    target = target_list[fig_idx]
    pred = pred_list[fig_idx]
    mask = mask_list[fig_idx]

    for class_idx, attribute_cam in enumerate(cam):
      ax = fig.add_subplot(num_rows, num_cols, class_idx+1)
      heatmapimg = heatmap_on_image(img, attribute_cam, 0.7)
      heatmapimg = heatmapimg.numpy().transpose(1, 2, 0)
      ax.imshow(heatmapimg)
      ax.set_title(label_str[class_idx])
      ax.axis('off')

    grid_img_file = f'{mode}{fig_idx}_{epoch}_grid.png'
    grid_img_path = os.path.join(log_dir, grid_img_file)

    try:
      target_string = label_list_to_str(target, label_str)
    except:
      print(target.shape)
      print(target)
    pred_string = label_list_to_str(pred, label_str)
    mask_string = label_list_to_str(mask, label_str)

    title = f'{mode}{fig_idx}_{epoch}\n'
    title += f'target: {target_string}\n'
    title += f'pred: {pred_string}\n'
    title += f'mask: {mask_string}\n'

    plt.suptitle(title)
    plt.savefig(grid_img_path)
    plt.close()


def log_cams(logger, 
             img_list, cam_list, target_list, pred_list, mask_list,
             epoch, mode, label_str, config=None):
  """
  Logging the cams
  """
  grid_cam_imgs = []
  num_classes = len(label_str)
  num_imgs = img_list.shape[0]

  # interpolate cams as same size of image
  cam_list = torch.nn.functional.interpolate(cam_list, size=img_list.shape[2:], mode='bicubic')

  for fig_idx in range(num_imgs):
    applied_imgs = []

    img = img_list[fig_idx]
    cam = cam_list[fig_idx]
    
    target = target_list[fig_idx]
    pred = pred_list[fig_idx]
    mask = mask_list[fig_idx]

    for attribute_cam in cam:
      applied_imgs.append(heatmap_on_image(img, attribute_cam, 0.7))

    # make grid
    img_grid = make_grid(applied_imgs, nrow=int(math.sqrt(num_classes)))
    grid_cam_imgs.append(img_grid)

    img_name = f'{mode}{fig_idx}_image'
    target_string = label_list_to_str(target, label_str)
    pred_string = label_list_to_str(pred, label_str)
    mask_string = label_list_to_str(mask, label_str)

    caption = f'{img_name}\n'
    caption += f'target: {target_string}\n'
    caption += f'pred: {pred_string}\n'
    caption += f'mask: {mask_string}\n'

    # log grid image
    if logger == 'wandb':
      img = wandb.Image(img_grid, caption=caption)
      wandb.log({img_name: [img]}, step=epoch)
    else:
      try:
        logger.add_image(img_name, img_grid, epoch)
      except:
        raise NotImplementedError