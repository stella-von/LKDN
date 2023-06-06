import argparse
import cv2
import glob
import numpy as np
import os.path as osp
import time
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='datasets/Set14/GTmod4', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results/ESRGAN', help='Path to restored images')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    args = parser.parse_args()

    # Configurations
    # -------------------------------------------------------------------------
    # folder_gt = 'datasets/celeba/celeba_512_validation'
    # folder_restored = 'datasets/celeba/celeba_512_validation_lq'
    folder_gt = args.gt
    folder_restored = args.restored
    # crop_border = 4
    suffix = args.suffix
    # -------------------------------------------------------------------------
    # RGB, normalized to [-1,1]
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()  # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # closer to "traditional" perceptual loss, when used for optimization
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + ext), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_alex(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.')
        lpips_all.append(lpips_val.item())
    end = time.perf_counter()
    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')
    print(f'Total time: {end - start:.2f} s, Average time: {(end - start) / len(lpips_all):.2f} s')


if __name__ == '__main__':
    main()
