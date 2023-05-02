import yaml
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

from attributedataset.datasets import get_dataloader
from models.modelutils import get_model

import utils.epochs as epochs
import utils.logging as logging

def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(config):
  # define device
  device = config['device']

  train_dataloader, test_dataloader, num_classes = get_dataloader(config)
  model = get_model(config, num_classes).to(device)
  writer = logging.logger_init(config)

  # define optimizer
  optimizer_config = config['OPTIMIZER']
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=optimizer_config['lr'],
      weight_decay=optimizer_config['weight_decay']
  )

  pbar = tqdm.tqdm(range(optimizer_config['epochs']))
  for epoch in pbar:
    train_loss = epochs.train_epoch(model, train_dataloader, optimizer, config, device)
    test_loss = epochs.test_epoch(model, test_dataloader, config, device)
    epochs.update_epoch(model, config)
      
    pbar.set_description(f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")
    logging.log(writer, train_loss, epoch, 'train', config)
    logging.log(writer, test_loss, epoch, 'test', config)

  logging.save_model(model, config)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    return parser.parse_args()


if __name__ == "__main__":
    # Config load
    args = argparser()
    config = load_config(args)
    print(config)

    main(config)