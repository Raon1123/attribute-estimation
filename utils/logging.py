import torch
from torch.utils.tensorboard import SummaryWriter
try:
  import wandb
except ImportError:
  wandb = None

def save_model(model, config):
  logging_config = config['LOGGING']
  torch.save(model.state_dict(), logging_config['model_path'])


def load_model(model, config):
  logging_config = config['LOGGING']
  model.load_state_dict(torch.load(logging_config['model_path']))
  return model


def logger_init(config):
  logging_config = config['LOGGING']

  if wandb is not None and logging_config['logger'] == 'wandb':
    wandb.init(
        project=logging_config['project'],
        name=logging_config['postfix'],
        config=config
    )
    wandb.watch_called = False
    writer = 'wandb'
  elif logging_config['logger'] == 'tensorboard':
    writer = SummaryWriter(logging_config['log_dir'])
  else:
    raise NotImplementedError
  
  return writer


def log(writer, loss, epoch, mode, config=None):
  if writer == 'wandb':
    wandb.log({f'{mode}_loss': loss, 'epoch': epoch})
  elif writer == 'tensorboard':
    writer.add_scalar(f'{mode}_loss', loss, epoch)
  else:
    raise NotImplementedError