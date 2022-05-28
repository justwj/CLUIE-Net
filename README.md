# Beyond Single Reference for Training: Underwater Image Enhancement via Comparative Learning
This repository includes two branches. This branch is the official PyTorch implementation of CLUIE-Net, another branch is the official PyTorch implementation of RQSD-Net.
## Dataset preparation 
You need to prepare datasets for following training and testing activities, the detailed information is at [Dataset Setup](data/README.md).

## Train
Before starting training，you should download the pretrained [VGG16](https://drive.google.com/file/d/1tnuhKbe70qk-VkmnRsHgVrku8lE4pIie/view?usp=sharing) model for compute the Content loss and the pretrained [RQSD-Net](https://drive.google.com/file/d/14JpdY4eciYTQQ5Wb4-_rgCZqnBQdOT9N/view?usp=sharing) model for compute the Superiority Discriminative loss, and then put them in "./data/vgg" and "./data/QC_ckpt".
``` 
python train.py --train_path /path_to_data
```
## Test
```
python test.py --test_path /path_to_data --fe_load_path /path_to_ckpt --fI_load_path /path_to_ckpt 
```
You can download the pretrained CLUIE-Net model from [here](https://drive.google.com/drive/folders/1uecaMgi3hqUy6PXIUUqAJaxkFNPLosAL?usp=sharing).


