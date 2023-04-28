# Preprocessing

This directory preprocess each data

## How to run preprocessing

### Common argument
- data_root: root directory of dataset

## Format

metadata
- `img_root`: root of image directory
- `label_str`: mapping label number to label name string (length: L)

each data
- `train_img_file`: filename for train image
- `val_img_file`: filename for validation image
- `label`: label for multi-label classification, each image has (1,L) shape. If value is 1, this image has positive label, 0 negative label, and -1 unknown label. 

## Example

```bash
python preprocess.py -D ~/dataset/RAP1 -S ~/dataset/attribute
```