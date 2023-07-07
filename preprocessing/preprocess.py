import argparse
import pickle
import os
import json

import numpy as np
from scipy.io import loadmat

DATASETS = ['rap1', 'pascal', 'coco']

def preprocess_rap1(args):
    data_len = 41585

    data_root = args.data_dir
    idx = args.i

    data = loadmat(os.path.join(data_root, 'RAP_annotation/RAP_annotation.mat'))
    data = data['RAP_annotation']

    img_root = os.path.join(data_root, 'RAP_dataset')
    img_file = [data[0][0][5][i][0][0] for i in range(data_len)]

    raw_attr_name = [data[0][0][3][i][0][0] for i in range(92)]
    raw_label = data[0][0][1]
    selected_attr_idx = np.array(range(51))[1:] # except 0: female

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

    # masking the label
    train_mask = get_masked_label(train_label, args.masking_rate)

    proc_dict = {
        'img_root': img_root,
        'label_str': attr_name,
        'train_img_file': train_img_file,
        'test_img_file': test_img_file,
        'train_label': train_label,
        'train_mask': train_mask,
        'test_label': test_label
    }

    return proc_dict


def preprocess_pascal(args):
    data_root = args.data_dir
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

    # masking the label
    train_mask = get_masked_label(label_matrix['train'], args.masking_rate)

    proc_dict = {
        'img_root': img_root,
        'label_str': list(catName_to_catID.keys()),
        'train_img_file': image_list['train'],
        'test_img_file': image_list['val'],
        'train_label': label_matrix['train'].astype(np.float16),
        'train_mask': train_mask,
        'test_label': label_matrix['val'].astype(np.float16)
    }

    return proc_dict


def preprocess_coco(args):
    data_root = args.data_dir

    json_train_path = os.path.join(data_root, 'annotations', 'instances_train2014.json')
    json_test_path = os.path.join(data_root, 'annotations', 'instances_val2014.json')

    with open(json_train_path, 'r') as f:
        json_train = json.load(f)
    with open(json_test_path, 'r') as f:
        json_test = json.load(f)

    # category
    category_list = []
    id_to_idx = {}
    category = json_train['categories']

    for i in range(len(category)):
        category_list.append(category[i]['name'])
        id_to_idx[category[i]['id']] = i
    
    # image id list
    train_img_id_list = []
    test_img_id_list = []

    train_img_id_list = sorted(np.unique([str(json_train['annotations'][i]['image_id']) for i in range(len(json_train['annotations']))]))
    test_img_id_list = sorted(np.unique([str(json_test['annotations'][i]['image_id']) for i in range(len(json_test['annotations']))]))

    train_img_id_list = np.array(train_img_id_list, dtype=np.int32)
    test_img_id_list = np.array(test_img_id_list, dtype=np.int32)

    train_image_id_to_index = {train_img_id_list[i]: i for i in range(len(train_img_id_list))}
    test_image_id_to_index = {test_img_id_list[i]: i for i in range(len(test_img_id_list))}

    num_categories = len(category_list)
    num_train_images = len(train_img_id_list)
    num_test_images = len(test_img_id_list)

    # label matrix
    train_label_matrix = np.zeros((num_train_images, num_categories))
    test_label_matrix = np.zeros((num_test_images, num_categories))

    train_image_ids = np.zeros(num_train_images)
    test_image_ids = np.zeros(num_test_images)

    for i in range(len(json_train['annotations'])):
        image_id = int(json_train['annotations'][i]['image_id'])
        row_index = train_image_id_to_index[image_id]

        category_id = int(json_train['annotations'][i]['category_id'])
        category_index = int(id_to_idx[category_id])

        train_label_matrix[row_index, category_index] = 1.0
        train_image_ids[row_index] = int(image_id)

    for i in range(len(json_test['annotations'])):
        image_id = int(json_test['annotations'][i]['image_id'])
        row_index = test_image_id_to_index[image_id]

        category_id = int(json_test['annotations'][i]['category_id'])
        category_index = int(id_to_idx[category_id])

        test_label_matrix[row_index, category_index] = 1.0
        test_image_ids[row_index] = int(image_id)

    # masking the label
    train_mask = get_masked_label(train_label_matrix, args.masking_rate)

    # image file name
    train_image = ['train2014/COCO_train2014_{:012d}.jpg'.format(int(train_image_ids[i])) for i in range(num_train_images)]
    test_image = ['val2014/COCO_val2014_{:012d}.jpg'.format(int(test_image_ids[i])) for i in range(num_test_images)]

    # save
    proc_dict = {
        'img_root': data_root,
        'label_str': category_list,
        'train_img_file': train_image,
        'test_img_file': test_image,
        'train_label': train_label_matrix.astype(np.float16),
        'train_mask': train_mask,
        'test_label': test_label_matrix.astype(np.float16)
    }

    return proc_dict


def get_masked_label(labels, masking_rate, masking_type='random'):
    """
    Generate masked label matrix

    Input
    - labels: label numpy matrix
    - masking_rate: masking rate, when -1.0, masking all labels except for one
    - masking_type: masking type, random or frequency

    Output
    - masked label matrix
    """
    num_instances, num_classes = labels.shape
    masked_labels = np.zeros((num_instances, num_classes))

    if masking_rate == 0.0:
        return masked_labels

    if masking_rate == -1.0:
        num_masked_labels = num_classes - 1
    else:
        num_masked_labels = int(num_classes * masking_rate)

    if masking_type != 'random':
        raise NotImplementedError
    else:
        # iterate per instance
        for mask, instance in zip(masked_labels, labels):
            mask_idx = np.random.choice(num_classes, num_masked_labels, replace=False)
            mask[mask_idx] = 1.0
    
    return masked_labels


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--masking_rate', type=float, default=-1.0)
    parser.add_argument('--masking_type', type=str, default='random', choices=['random', 'frequency']) # TODO add maksing type
    
    args = parser.parse_args()
    return args

# Testing todo
# is labels in ours range -1 0 1?

if __name__ == "__main__":
    args = argparser()

    pkl_file = '_'.join([args.dataset, str(args.masking_rate), str(args.seed),'preprocess.pkl'])
    save_path = os.path.join(args.save_dir, pkl_file)
    # check if save path exists
    if os.path.exists(save_path) and not args.force:
        raise ValueError('Save path already exists: {}. If you generate even exist, please use --force option'.format(save_path))

    # set seed
    np.random.seed(args.seed)

    if args.dataset == 'rap1':
        proc_dict = preprocess_rap1(args)
    elif args.dataset == 'pascal':
        proc_dict = preprocess_pascal(args)
    elif args.dataset == 'coco':
        proc_dict = preprocess_coco(args)
    else:
        raise NotImplementedError
    
    # saving proc_dict
    with open(save_path, 'wb') as f:
        pickle.dump(proc_dict, f)
    
    print('Preprocessing done!')
    print('Saved at {}'.format(save_path))
    print('Dataset: {}'.format(args.dataset))
    print('Masking rate: {}'.format(args.masking_rate))
    print('Number of train images: {}'.format(len(proc_dict['train_img_file'])))
    print('Number of test images: {}'.format(len(proc_dict['test_img_file'])))
    print('Number of attributes: {}'.format(len(proc_dict['label_str'])))