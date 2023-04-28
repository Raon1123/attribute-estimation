import argparse
import pickle
import os

import numpy as np
from scipy.io import loadmat

def preprocess_rap1(args):
    data_len = 41585

    data_root = args.data_dir
    save_root = args.save_dir
    idx = args.i

    data = loadmat(os.path.join(data_root, 'RAP_annotation/RAP_annotation.mat'))
    data = data['RAP_annotation']

    img_root = os.path.join(data_root, 'RAP_dataset')
    img_file = [data[0][0][5][i][0][0] for i in range(data_len)]

    raw_attr_name = [data[0][0][3][i][0][0] for i in range(92)]
    raw_label = data[0][0][1]
    selected_attr_idx = np.array(range(51))

    label = raw_label[:, selected_attr_idx].astype(np.float16)
    attr_name = [raw_attr_name[i] for i in selected_attr_idx]

    train_img_file = []
    test_img_file = []

    train = data[0][0][0][idx][0][0][0][0][0, :] - 1
    test = data[0][0][0][idx][0][0][0][1][0, :] - 1

    train_img_file.append([img_file[i] for i in train])
    test_img_file.append([img_file[i] for i in test])

    train_label = label[train, :]
    test_label = label[test, :]

    proc_dict = {
        'img_root': img_root,
        'label_str': attr_name,
        'train_img_file': train_img_file,
        'test_img_file': test_img_file,
        'train_label': train_label,
        'test_label': test_label
    }

    with open(os.path.join(save_root, 'RAPv1.pkl'), 'wb') as f:
        pickle.dump(proc_dict, f)


def argparser():
    parser = argparse.ArgumentParser(
        prog='RAPv1 dataset preprocessing',
        description='preprocess RAPv1 dataset for our framework'
    )

    parser.add_argument('-D', '--dataset', type=str, default='RAPv1', required=True)
    parser.add_argument('-R', '--data_dir', type=str, required=True,
                        help='root directory of raw dataset')
    parser.add_argument('-S', '--save_dir', type=str, required=True,
                        help='save preprocessed results')
    parser.add_argument('-i', type=int, default=0, choices=range(5),
                        help='index of RAPv1 dataset')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # root_dir ~/dataset/RAP1
    args = argparser()
    if args.dataset == 'RAPv1':
        preprocess_rap1(args)
    else:
        raise NotImplementedError