# Beyond Single Reference for Training: Underwater Image Enhancement via Comparative Learning
This repository is the official PyTorch implementation of CLUIE-Net.
## Dataset preparation 
You need to prepare datasets for following training and testing activities, the detailed information is at [Dataset Setup](data/README.md).

## Train
``` 
python train.py --train_path /path_to_data
```
## Test
```
python test.py --test_path /path_to_data --fe_load_path /p
```
You can download the trained model from [here](https://drive.google.com/file/d/1vbY4GZ5-AwVKouDFHvFj9nL-grnIB2d3/view?usp=sharing).

## Citation
```
@article{qi2022sguie,
  title={SGUIE-Net: Semantic Attention Guided Underwater Image Enhancement with Multi-Scale Perception},
  author={Qi, Qi and Li, Kunqian and Zheng, Haiyong and Gao, Xiang and Hou, Guojia and Sun, Kun},
  journal={arXiv preprint arXiv:2201.02832},
  year={2022}
}
```

## Acknowledgements
- https://github.com/xahidbuffon/SUIM
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
