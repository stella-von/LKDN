import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.lkdns_arch import LKDN_S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/LKDN/LKDN-S_del_rep_x4.pth'  # noqa: E501
    )
    parser.add_argument(
        '--input', type=str, default='datasets/DIV2K/DIV2K_valid_LR_bicubic/X4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/LKDN-S', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = LKDN_S(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=42,
        num_atten=42,
        num_block=5,
        upscale=4,
        num_in=4,
        conv='BSConvU',
        upsampler='pixelshuffledirect')
    model.load_state_dict(torch.load(args.model_path), strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx + 1, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)


if __name__ == '__main__':
    main()
