## Data Preparation
### 1. Download the regional quality-superiority dataset for underwater images [RQSD-UI](https://drive.google.com/drive/folders/1gA3Ic7yOSbHd3w214-AgMI9UleAt4bRM?usp=sharing).

### 2. The structure of data folder is as follows:
```
├── dataset_name
    ├── E_imgs
        ├── d_r_1_CLAHE.jpg
        ├── d_r_1_DCP.jpg
        └── ...
    ├── train-validation_imgs
        ├── d_r_3_.jpg
        ├── d_r_5_.jpg
        └── ...
    ├── test_imgs
        ├── d_r_1_.jpg
        ├── d_r_4_.jpg
        └── ...
    ├── train-validation_pairs
        ├── train-Cons.txt
        ├── train-Cons-L1.txt
        └── ...
    ├── test_pairs
        ├── test-Cons.txt
        ├── test-Cons-L1.txt
        └── ...
    ├── train-validation-GT-QSmaps
        ├── 32
            ├── w_r_299_ULAP-w_r_299_FUSION-w_r_299_FUSION.npy
            ├── w_r_299_ULAP-w_r_299_CLAHE-w_r_299_FUSION.npy
            └── ...
        ├── 16
            ├── w_r_299_ULAP-w_r_299_FUSION-w_r_299_FUSION.npy
            ├── w_r_299_ULAP-w_r_299_CLAHE-w_r_299_FUSION.npy
            └── ...
        ├── 8
            ├── w_r_299_ULAP-w_r_299_FUSION-w_r_299_FUSION.npy
            ├── w_r_299_ULAP-w_r_299_CLAHE-w_r_299_FUSION.npy
            └── ...    
    ├── test-GT-QSmaps
        ├── test-Cons-GT-QSmaps
            ├── d_r_1_CLAHE-d_r_1_GCHE-d_r_1_GCHE.npy
            ├── d_r_1_DCP-d_r_1_GCHE-d_r_1_GCHE.npy
            └── ...
        ├── test-Cons-L1-GT-QSmaps
            ├── d_r_26_CLAHE-d_r_26_DIVE-d_r_26_GCHE.npy
            ├── d_r_26_CLAHE-d_r_26_RETINEX-d_r_26_GCHE.npy
            └── ...
        └── ...
    ├── labelmap
        ├── w_r_299_ULAP-w_r_299_FUSION-w_r_299_FUSION.npy
        ├── w_r_299_ULAP-w_r_299_CLAHE-w_r_299_FUSION.npy


```


