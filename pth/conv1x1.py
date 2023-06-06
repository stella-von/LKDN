import torch
import torch.nn.functional as F

input = torch.randn(64, 42, 64, 64)

k3x3 = torch.randn(42, 1, 3, 3)
b3x3 = torch.randn(42)

k1x1 = torch.randn(42, 1, 1, 1)
b1x1 = torch.randn(42)

# conv-3x3
v1 = F.conv2d(input, k3x3, b3x3, padding=1, groups=42)
# conv-1x1
v2 = F.conv2d(input, k1x1, b1x1, padding=0, groups=42)
v3 = v1 + v2

print(v1.shape)
print(v2.shape)

# re-param conv kernel
r3x3 = F.pad(k1x1, (1, 1, 1, 1), 'constant', 0.0) + k3x3
# re-param conv bias
rb = b1x1 + b3x3
print(r3x3.shape)

v4 = F.conv2d(input, r3x3, rb, padding=1, groups=42)

# compare
print(v3 - v4)
print(torch.sum(v4 - v3))
