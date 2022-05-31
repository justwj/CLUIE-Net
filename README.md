# RQSD-Net
This branch is the official PyTorch implementation of **RQSD-Net**.
## Dataset preparation 
You need to prepare datasets for following training and testing activities, the detailed information is at [Dataset Setup](data/readme.md).

## Train
``` 
python train.py --txt_path /path_to_data
```
## Test
```
python test.py  --txt_path /path_to_data --gt_path /path_to_data --modelsave_path /path_to_checkpoint
```
You can download the pretrained **RQSD-Net** model from [here](https://drive.google.com/file/d/14JpdY4eciYTQQ5Wb4-_rgCZqnBQdOT9N/view?usp=sharing).


## Acknowledgements
- https://github.com/trentqq/SUIM-E

