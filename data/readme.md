## Data Preparation
### 1. Download the regional quality-superiority dataset for underwater images [RQSD-UI](https://drive.google.com/drive/folders/1gA3Ic7yOSbHd3w214-AgMI9UleAt4bRM?usp=sharing).

### 2. The structure of data folder is as follows:
```
├── dataset_name
    ├── E_img
        ├── d_r_1_CLAHE.jpg
        ├── d_r_1_DCP.jpg
        └── ...
    ├── train
        ├── d_r_3_.jpg
        ├── d_r_5_.jpg
        └── ...
    ├── test
        ├── d_r_1_.jpg
        ├── d_r_4_.jpg
        └── ...
    ├── train_txt
        ├── train-easy.txt
        ├── train-tough1.txt
        └── ...
    ├── test_txt
        ├── test-easy.txt
        ├── test-tough1.txt
        └── ...
    ├── test_labelmap
        ├── test-easy_label
            ├── d_r_1_CLAHE-d_r_1_GCHE-d_r_1_GCHE.npy
            ├── d_r_1_DCP-d_r_1_GCHE-d_r_1_GCHE.npy
            └── ...
        ├── test-tough_label
            ├── d_r_26_CLAHE-d_r_26_DIVE-d_r_26_GCHE.npy
            ├── d_r_26_CLAHE-d_r_26_RETINEX-d_r_26_GCHE.npy
            └── ...
        └── ...
    ├── test_labelmap
        ├── test-easy_label
            ├── d_r_1_CLAHE-d_r_1_GCHE-d_r_1_GCHE.npy
            ├── d_r_1_DCP-d_r_1_GCHE-d_r_1_GCHE.npy
            └── ...
        ├── test-tough_label
            ├── d_r_26_CLAHE-d_r_26_DIVE-d_r_26_GCHE.npy
            ├── d_r_26_CLAHE-d_r_26_RETINEX-d_r_26_GCHE.npy
            └── ...
        └── ...


```

## Custom Datasets(without segmentation map)
For other datasets without ground truth segmentation map, users can choose to:
1. Manually mark the segmentation map
2. Using [SUIM-Net](https://github.com/xahidbuffon/SUIM) to generate predicted semantic segmentation, by loading [pre-trained models](https://drive.google.com/drive/folders/1aoluekvB_CzoaqGhLutwtJptIOBasl7i).

Then, organize the obtained segmentation map as described [above](#Data-Preparation).

