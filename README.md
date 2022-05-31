# Beyond Single Reference for Training: Underwater Image Enhancement via Comparative Learning
This repository includes two branches. This branch is the official PyTorch implementation of **CLUIE-Net**, another branch is the official PyTorch implementation of **RQSD-Net**.
## Test_Demo
You should download the pretrained **CLUIE-Net** model from [here](https://drive.google.com/drive/folders/1uecaMgi3hqUy6PXIUUqAJaxkFNPLosAL?usp=sharing),then you should put them in the folder  **'./ckpt'**.
```
python test.py --fe_load_path /path_to_ckpt --fI_load_path /path_to_ckpt 
```
You can find the enhanced results in folder **'./output'**

