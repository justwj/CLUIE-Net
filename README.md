# Beyond Single Reference for Training: Underwater Image Enhancement via Comparative Learning
This repository is the official PyTorch implementation of CLUIE-Net.
## Dataset preparation 
You need to prepare datasets for following training and testing activities, the detailed information is at [Dataset Setup](data/README.md).

## Train
Before starting training，you should download the pretrained [VGG16 model]() for compute the Content loss and the pretrained [RQSD-Net model]() for compute the Superiority Discriminative loss, then put them in data/vgg and data/QC_ckpt.
``` 
python train.py --train_path /path_to_data
```
## Test
```
python test.py --test_path /path_to_data --fe_load_path /path_to_ckpt --fI_load_path /path_to_ckpt 
```
You can download the pretrained CLUIE-Net model from [here](https://drive.google.com/drive/folders/1uecaMgi3hqUy6PXIUUqAJaxkFNPLosAL?usp=sharing).

## Acknowledgements
- https://github.com/xahidbuffon/SUIM

