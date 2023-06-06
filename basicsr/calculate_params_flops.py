import torch
from archs.lkdn_arch import LKDN as net
# from archs.lkdns_arch import LKDN_S as net
from thop import profile
# pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

# 将@ARCH_REGISTRY.register()注释
model = net()

net_cls_str = f'{model.__class__.__name__}'

# thop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = torch.randn(1, 3, 320, 180).to(device)
flops, params = profile(model, (inputs, ))
print(f'Network: {net_cls_str}, with flops(1280 x 720): {flops/1e9:.2f} GMac, with active parameters: {params/1e3} K.')
