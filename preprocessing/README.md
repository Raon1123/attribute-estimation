# Preprocessing

This directory preprocess each data

## How to run preprocessing

### Common argument

Required
- `--dataset, -D`: type of dataset, choice of `['rap1', 'pascal', 'coco']`
- `--data_dir, -R`: root directory path of raw dataset
- `--save_dir, -S`: save directory path of preprocesed results

Optional
- `--seed`: generation seed 
- `--force`: generate even file exist
- `--unmasking_rate`: masking ratio of data
- Special: `-i` index for RAPv1 dataset

## Format

metadata
- `img_root`: root of image directory
- `label_str`: mapping label number to label name string (length: L)

each data
- `[train, test]_img_file`: filename for [train, test] image
- `[train, test]_label`: label for multi-label classification, each image has (1,L) shape. If value is 1, this image has positive label, 0 unknown label, and -1 negative label. 
- `train_mask`: mask for train, if value is 1 the label will mask (partial label condition)

## Example

```bash
python preprocess.py -D rap1 -R ~/dataset/RAP1 -S ~/dataset/attribute
```