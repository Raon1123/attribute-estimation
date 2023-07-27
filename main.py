# Description: Main script for training and testing

import tqdm

from attributedataset.datasetutils import get_dataloader

import models.modelutils as modelutils
import utils.epochs as epochs
import utils.logging as logging
import utils.parsing as parsing

try:
    import wandb
except ImportError:
    wandb = None


def main(config):
    # define device
    device = config['device']
    use_feature = parsing.get_use_feature(config)
    log_interval, save_imgs = parsing.get_log_configs(config)
    pkl_list = parsing.get_pkl_list(config)

    for pkl_file in pkl_list:
        config['DATASET']['pkl_file'] = pkl_file

        train_dataloader, test_dataloader, meta_info = get_dataloader(config)
        num_classes = meta_info['num_classes']
        
        experiment(train_dataloader, test_dataloader, num_classes,
                    log_interval=log_interval, save_imgs=save_imgs,
                    use_feature=use_feature, device=device)
        
        if wandb is not None:
            wandb.finish()
        
        
def experiment(train_dataloader, 
               test_dataloader, 
               num_classes,
               log_interval=1,
               save_imgs=1,
               use_feature=False,
               device='cpu'):
    # define model and logger
    model = modelutils.get_model(config, num_classes, use_feature=use_feature).to(device)
    logger = logging.logger_init(config)

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

        # logging loss
        pbar.set_description(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")
        loss_dict = {'train_loss': train_loss, 'test_loss': test_loss}
        logging.log_loss(logger, loss_dict, epoch, config)

        # logging metrics
        train_metrics = epochs.evaluate_result(
            model, train_dataloader, epoch, config, device, masking=True)
        test_metrics = epochs.evaluate_result(
            model, test_dataloader, epoch, config, device, masking=False, prefix='test_')
        
        logging.log_metrics(logger, train_metrics, epoch, config)
        logging.log_metrics(logger, test_metrics, epoch, config)

        # logging cams
        if (epoch + 1) % log_interval == 0:
            # train cams
            imgs, cams = epochs.evaluate_cam(model, train_dataloader, num_imgs=save_imgs, device=device)
            imgs, cams = imgs[:save_imgs], cams[:save_imgs] # (save_imgs, num_classes, H, W), (save_imgs, num_classes, 7, 7)
            logging.write_cams(config, imgs, cams, epoch, mode='train')
            logging.log_cams(logger, imgs, cams, epoch, mode='train', config=config)

            # test cams
            imgs, cams = epochs.evaluate_cam(model, test_dataloader, num_imgs=save_imgs, device=device)
            imgs, cams = imgs[:save_imgs], cams[:save_imgs]
            logging.write_cams(config, imgs, cams, epoch, mode='test')
            logging.log_cams(logger, imgs, cams, epoch, mode='test', config=config)

    metrics = epochs.evaluate_result(
        model, test_dataloader, epoch, config, device, 
        saving=True, masking=True)
    logging.print_metrics(metrics)
    logging.save_model(model, config)


if __name__ == "__main__":
    # Config load
    args = parsing.argparser()
    config = parsing.load_config(args)
    print(config)

    if args.feature:
        from attributedataset.datasetutils import generate_feature
        generate_feature(config)
    
    main(config)
    exit(0)
