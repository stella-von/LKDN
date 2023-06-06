import torch
import torch.nn.functional as F

input = torch.randn(64, 42, 64, 64)

k1x1 = torch.randn(84, 42, 1, 1)
b1x1 = torch.randn(84)

k3x3 = torch.randn(42, 84, 3, 3)
b3x3 = torch.randn(42)

# conv-1x1
v1 = F.conv2d(input, k1x1, b1x1, padding=0)
v1 = F.pad(v1, (1, 1, 1, 1), 'constant', 0)
# explicitly padding with bias
b0_pad = b1x1.view(1, -1, 1, 1)
v1[:, :, 0:1, :] = b0_pad
v1[:, :, -1:, :] = b0_pad
v1[:, :, :, 0:1] = b0_pad
v1[:, :, :, -1:] = b0_pad
# conv-3x3
v2 = F.conv2d(v1, k3x3, b3x3)

print(v1.shape)
print(v2.shape)

# re-param conv kernel
r1x1 = k1x1.permute(1, 0, 2, 3)
r3x3 = F.conv2d(k3x3, r1x1)
print(r3x3.shape)

# re-param conv bias
rb = torch.ones(1, 84, 3, 3) * b1x1.view(1, -1, 1, 1)  # 1 84 3 3
rb = F.conv2d(rb, k3x3).view(-1, ) + b3x3

v3 = F.conv2d(input, r3x3, rb, padding=1)

# compare
print(v3 - v2)
print(torch.sum(v3 - v2))
