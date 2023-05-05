import yaml
import tqdm
import argparse

import torch

from attributedataset.datasetutils import get_dataloader

import models.modelutils as modelutils
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
    model = modelutils.get_model(config, num_classes).to(device)
    writer = logging.logger_init(config)

    # define optimizer
    optimizer_config = config['OPTIMIZER']
    optimizer = modelutils.get_optimizer(model, config)
    scheduler = modelutils.get_scheduler(optimizer, config)

    pbar = tqdm.tqdm(range(optimizer_config['epochs']))
    for epoch in pbar:
        train_loss = epochs.train_epoch(
            model, train_dataloader, optimizer, config, device
        )
        test_loss = epochs.test_epoch(model, test_dataloader, config, device)

        if scheduler is not None:
            scheduler.step()

        epochs.update_epoch(model, config)

        pbar.set_description(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")
        logging.log(writer, train_loss, epoch, 'train', config)
        logging.log(writer, test_loss, epoch, 'test', config)

        metrics = epochs.evaluate_result(
            model, test_dataloader, epoch, config, writer, device)
        logging.log_metrics(writer, metrics, epoch, config)

    logging.save_model(model, config)
    metrics = epochs.evaluate_result(
        model, test_dataloader, epoch, config, writer, device)
    
    # write to text
    with open('result.txt', 'w') as f:
        f.write(str(metrics))


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--feature', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    # Config load
    args = argparser()
    
    if args.feature:
        from attributedataset.datasetutils import generate_feature
        generate_feature(args.config)
        exit(0)
    
    config = load_config(args)
    print(config)

    main(config)
