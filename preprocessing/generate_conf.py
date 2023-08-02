import argparse
import os
import yaml

from preprocess import DATASETS

def parse_args():
    parser = argparse.ArgumentParser(description='Generate config file')
    parser.add_argument('--project_name', type=str, default='pascal', help='project name')
    parser.add_argument('-D', '--dataset', type=str, choices=DATASETS, required=True)
    args = parser.parse_args()
    return args


def write_conf(args, project_name, masking_ratio, lr, wd):
    dataset = args.dataset

    postfixs = [f'msk{masking_ratio}',
                f'lr{lr}',
                f'wd{wd}']
    postfixs = '_'.join(postfixs)

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', dataset)
    os.makedirs(config_dir, exist_ok=True)
    
    config_name = f'{dataset}_{postfixs}.yaml'

    logging_config = {
        'logger': 'wandb',
        'log_dir': './logs',
        'model_path': './save',
        'log_interval': 5,
        'save_imgs': 10,
        'load_model': False,
        'project': project_name,
        'postfix': postfixs
    }

    loader_config = {
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True
    }

    method_config = {
        'name': 'BoostCAM',
        'backbone': 'resnet18',
        'freeze_backbone': False,
        'mod_scheme': 'LL-R',
        'delta_rel': 0.5,
        'alpha': 5.0
    }

    scheduler_config = {
        'name': 'StepLR',
        'step_size': 10,
        'gamma': 0.1
    }

    optimizer_config = {
        'name': 'Adam',
        'lr': 1e-5,
        'weight_decay': 1e-4,
        'epochs': 30,
        'scheduler': scheduler_config
    }

    pkl_list = []
    seeds = list(range(5))
    for seed in seeds:
        pkl_file = [dataset,
                    masking_ratio,
                    seed,
                    'preprocess.pkl']
        pkl_file = '_'.join([str(x) for x in pkl_file])
        pkl_list.append(pkl_file)

        if masking_ratio == 0.0:
            break

    dataset_config = {
        'name': 'pascal',
        'pkl_root': '/share_home/slurmayp/dataset/attribute/',
        'pkl_list': pkl_list,
        'use_feature': False,
        'transforms': {
            'input_size': 224,
            'input_ratio': 1.0,
        }
    }

    config = {
        'device': 'cuda:0',
        'LOGGING': logging_config,
        'DATALOADER': loader_config,
        'METHOD': method_config,
        'OPTIMIZER': optimizer_config,
        'DATASET': dataset_config
    }

    # write config
    config_PATH = os.path.join(config_dir, config_name)
    with open(config_PATH, 'w') as f:
        yaml.dump(config, f)

    # write script
    conf_dir = f'../scripts/{dataset}'
    os.makedirs(conf_dir, exist_ok=True)

    # slurm script
    slurm_script_list = [
        '#!/bin/bash',
        f'#SBATCH --job-name=BoostCAM_{dataset}',
        '#SBATCH --output=logs/%x-%j.out',
        '#SBATCH --error=logs/%x-%j.err',
        '#SBATCH --nodes=1',
        '#SBATCH -p compute1',
        '#SBATCH --nodelist=unistml8',
        '#SBATCH --gres=gpu:rtx3090:1',
    ]

    slurm_script = '\n'.join(slurm_script_list)

    config_PATH = os.path.join('configs', dataset, config_name)

    start_script = [
        '. /usr/share/modules/init/sh',
        'module purge',
        'module load conda',
        'source $CONDA_HOME/etc/profile.d/conda.sh',
        'conda activate pytorch',
        f'python main.py --config {config_PATH}'
    ]
    start_script = '\n'.join(start_script)

    script = '\n'.join([slurm_script, start_script])

    script_PATH = os.path.join(conf_dir, f'{dataset}_{postfixs}.sh')
    with open(script_PATH, 'w') as f:
        f.write(script)


def main():
    args = parse_args()

    project_name = 'BoostCAM18_FullActivate'
    masking_ratios = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    for masking_ratio in masking_ratios:
        lr = 1e-5
        wd = 1e-4
        write_conf(args, project_name, masking_ratio, lr, wd)
        wd = 0.0
        write_conf(args, project_name, masking_ratio, lr, wd)


if __name__ == '__main__':
    main()