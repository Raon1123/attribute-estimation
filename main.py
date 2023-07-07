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
    try:
        use_feature = config['DATASET']['use_feature']
    except:
        use_feature = False

    train_dataloader, test_dataloader, num_classes = get_dataloader(config)
    model = modelutils.get_model(config, num_classes, use_feature=use_feature).to(device)
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
        logging.log_loss(writer, train_loss, epoch, 'train', config)
        logging.log_loss(writer, test_loss, epoch, 'test', config)

        msk_metrics = epochs.evaluate_result(
            model, train_dataloader, epoch, config, device, masking=True)
        metrics = epochs.evaluate_result(
            model, train_dataloader, epoch, config, device, masking=False)
        
        logging.log_metrics(writer, metrics, epoch, config)
        logging.log_metrics(writer, msk_metrics, epoch, config)

    metrics = epochs.evaluate_result(
        model, test_dataloader, epoch, config, device, 
        saving=True, masking=True)
    logging.save_model(model, config)
    

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--feature', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    # Config load
    args = argparser()
    config = load_config(args)
    print(config)

    if args.feature:
        from attributedataset.datasetutils import generate_feature
        generate_feature(config)
    
    main(config)

    exit(0)
