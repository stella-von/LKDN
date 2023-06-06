# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0

import glob
import numpy as np
# import os
import os.path as osp
# import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class L2pooling(nn.Module):

    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class DISTS(torch.nn.Module):

    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)
        if load_weights:
            # weights = torch.load(os.path.join(sys.prefix, 'weights.pt'))
            weights = torch.load('/root/autodl-tmp/BasicSR/scripts/metrics/DISTS_pytorch/weights.pt')
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean)**2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean)**2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score


def prepare_image(image, resize=True):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


def main():
    start = time.perf_counter()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DISTS().to(device)

    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='datasets/Set14/GTmod12', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results/ESRGAN', help='Path to restored images')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    args = parser.parse_args()

    folder_gt = args.gt
    folder_restored = args.restored
    suffix = args.suffix

    dists_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        ref = prepare_image(Image.open(img_path).convert("RGB"), resize=False)
        dist = prepare_image(Image.open(osp.join(folder_restored, basename + suffix + ext)).convert("RGB"),
                             resize=False)
        assert ref.shape == dist.shape

        ref = ref.to(device)
        dist = dist.to(device)
        # calculate dists
        dists_val = model(ref, dist)
        print(f'{i+1:3d}: {basename:25}. \tDISTS: {dists_val.item():.6f}.')

        dists_all.append(dists_val.item())
    end = time.perf_counter()
    print(f'Average: DISTS: {sum(dists_all) / len(dists_all):.6f}')
    print(f'Total time: {end - start:.2f} s, Average time: {(end - start) / len(dists_all):.2f} s')


if __name__ == '__main__':
    main()
