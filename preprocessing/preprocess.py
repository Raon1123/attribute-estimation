import argparse
import pickle
import os

import numpy as np
from scipy.io import loadmat

DATASETS = ['rap1', 'pascal']

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
    selected_attr_idx = np.array(range(51))[1:]

    label = raw_label[:, selected_attr_idx].astype(np.float16)
    attr_name = [raw_attr_name[i] for i in selected_attr_idx] 

    train_img_file = []
    test_img_file = []

    train = data[0][0][0][idx][0][0][0][0][0, :] - 1
    test = data[0][0][0][idx][0][0][0][1][0, :] - 1

    train_img_file.extend([img_file[i] for i in train])
    test_img_file.extend([img_file[i] for i in test])

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


def preprocess_pascal(args):
    data_root = args.data_dir
    save_root = args.save_dir
    img_root = os.path.join(data_root, 'VOCdevkit', 'VOC2012', 'JPEGImages')

    catName_to_catID = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }

    catID_to_catName = {catName_to_catID[k]: k for k in catName_to_catID}

    ann_dict = {}
    image_list = {'train': [], 'val': []}
    label_matrix = {}

    for phase in ['train', 'val']:
        for cat in catName_to_catID:
            load_file = os.path.join(data_root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', str(cat) + '_' + phase + '.txt')
            with open(load_file, 'r') as f:
                for line in f:
                    cur_line = line.rstrip().split(' ')
                    image_id = cur_line[0]
                    label = cur_line[-1]
                    img_file = image_id + '.jpg'
                    if int(label) == 1:
                        if img_file not in ann_dict:
                            ann_dict[img_file] = []
                            image_list[phase].append(img_file)
                        ann_dict[img_file].append(catName_to_catID[cat])

        image_list[phase].sort()
        num_imgs = len(image_list[phase])
        label_matrix[phase] = np.zeros((num_imgs, len(catName_to_catID)))
        for i in range(num_imgs):
            cur_img = image_list[phase][i]
            label_indicies = np.array(ann_dict[cur_img])
            label_matrix[phase][i, label_indicies] = 1.0

    proc_dict = {
        'img_root': img_root,
        'label_str': list(catName_to_catID.keys()),
        'train_img_file': image_list['train'],
        'test_img_file': image_list['val'],
        'train_label': label_matrix['train'].astype(np.float16),
        'test_label': label_matrix['val'].astype(np.float16)
    }

    with open(os.path.join(save_root, 'PASCAL.pkl'), 'wb') as f:
        pickle.dump(proc_dict, f)
                    

def argparser():
    parser = argparse.ArgumentParser(
        prog='attribute dataset preprocessing',
        description='preprocess attribute dataset for our framework'
    )

    parser.add_argument('-D', '--dataset', type=str, choices=DATASETS, required=True)
    parser.add_argument('-R', '--data_dir', type=str, required=True,
                        help='root directory of raw dataset')
    parser.add_argument('-S', '--save_dir', type=str, required=True,
                        help='save preprocessed results')
    parser.add_argument('-i', type=int, default=0, choices=range(5),
                        help='index of RAPv1 dataset')
    
    args = parser.parse_args()
    return args

# Testing todo
# is labels in ours range -1 0 1?


if __name__ == "__main__":
    # root_dir ~/dataset/RAP1
    args = argparser()
    if args.dataset == 'rap1':
        preprocess_rap1(args)
    elif args.dataset == 'pascal':
        preprocess_pascal(args)
    else:
        raise NotImplementedError