# LKDN
[Large Kernel Distillation Network for Efficient Single Image Super-Resolution](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Xie_Large_Kernel_Distillation_Network_for_Efficient_Single_Image_Super-Resolution_CVPRW_2023_paper.pdf)

Chengxing Xie, Xiaoming Zhang, Linze Li, Haiteng Meng, Tianlin Zhang, Tianrui Li and Xiaole Zhao

## Environment

- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend >= 1.11)**
- [BasicSR = 1.4.2](https://github.com/XPixelGroup/BasicSR)

### Installation

```
pip install -r requirements.txt
python setup.py develop
```

## How To Test

- Refer to `./options/test/LKDN` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.
- The pretrained models are available in `./experiments/pretrained_models/LKDN`.
- Then run the follwing codes (taking `LKDN_x4.pth` as an example):

```
python basicsr/test.py -opt options/test/LKDN/test_LKDN_x4.yml
```

The testing results will be saved in the `./results` folder.

- Refer to `./inference` for **inference** without the ground truth image.
- Refer to `./basicsr/calculate_params_flops.py` for calculating the **parameters and flops.**

## How To Train

- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).
- The training command is like:

```
python basicsr/train.py -opt options/train/LKDN/train_LKDN_x4.yml
```

More training commands can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md).

The training logs and weights will be saved in the `./experiments` folder.

## How To Re-parameterize

Refer to `./pth` for the validation and use of re-parameterization.

`conv1x1_3x3.py`, `conv1x1.py` and `shortcut.py` respectively **verify** the three re-parameterization methods.

`del_params_ema.py` simplifies the `.pth` file. (Remove the additional parameters retained when using EMA.)

`print_pth.py` prints the content of the `.pth` file.

`reparm.py` reparameterizes the model.

**Note that the `LKDN-S_del_rep_x4.pth` is the model after re-parameterizing, and the `LKDN-S_x4.pth` is the model without re-parameterizing.**

## Results

**Benchmark results on SR ×4. Multi-Adds is calculated with a 1280 × 720 GT image.**

| **Method** | **Params[K]** | **Multi-Adds[G]** | **Set5 PSNR/SSIM** | **Set14 PSNR/SSIM** | **BSD100 PSNR/SSIM** | **Urban100 PSNR/SSIM** | **Manga109 PNSR/SSIM** |
| :--------: | :-----------: | :---------------: | :----------------: | :-----------------: | :------------------: | :--------------------: | :--------------------: |
|    BSRN    |      352      |       19.4        |    32.35/0.8966    |    28.73/0.7847     |     27.65/0.7387     |      26.27/0.7908      |      30.84/0.9123      |
|   VapSR    |      342      |       19.5        |    32.38/0.8978    |    28.77/0.7852     |     27.68/0.7398     |      26.35/0.7941      |      30.89/0.9132      |
|    LKDN    |      322      |       18.3        |    32.39/0.8979    |    28.79/0.7859     |     27.69/0.7402     |      26.42/0.7965      |      30.97/0.9140      |

The inference results on benchmark datasets are available at [Google Drive](https://drive.google.com/drive/folders/18If6wTJEU1Xpqf7uDbSIMO2PzQCu7kN_?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1A0vFDCqD7dhs3O3P3_0jkw?pwd=lkdn).

## Contact

If you have any question, please email zxc0074869@gmail.com.
