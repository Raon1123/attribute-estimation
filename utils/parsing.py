import yaml
import argparse

def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--feature', action='store_true')
    return parser.parse_args()


def get_use_feature(config):
    try:
        use_feature = config['DATASET']['use_feature']
    except:
        use_feature = False

    return use_feature


def get_log_configs(config):
    try:
        log_interval = config['LOGGING']['log_interval']
        save_imgs = config['LOGGING']['save_imgs']
    except:
        log_interval = 1
        save_imgs = 1

    return log_interval, save_imgs


def get_pkl_list(config):
    try:
        pkl_list = config['DATASET']['pkl_list']
    except:
        pkl_list = [config['DATASET']['pkl_file']]

    return pkl_list