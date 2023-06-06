import torch
from torch import nn as nn

from basicsr.archs.lkdn_blocks import LKDB, BSConvU, BSConvU_idt, BSConvU_rep, UpsampleOneStep, Upsampler_rep
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class LKDN_S(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=42,
                 num_atten=42,
                 num_block=5,
                 upscale=4,
                 num_in=4,
                 conv='BSConvU',
                 upsampler='pixelshuffledirect'):
        super().__init__()
        self.num_in = num_in
        if conv == 'BSConvU_idt':
            self.conv = BSConvU_idt
        elif conv == 'BSConvU_rep':
            self.conv = BSConvU_rep
        elif conv == 'BSConvU':
            self.conv = BSConvU
        else:
            raise NotImplementedError(f'conv {conv} is not supported yet.')
        print(conv)
        self.fea_conv = BSConvU(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)

        self.B1 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B2 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B3 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B4 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B5 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle_rep':
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        input = torch.cat([input] * self.num_in, dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output
