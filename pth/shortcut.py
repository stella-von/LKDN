import torch
import torch.nn.functional as F

input = torch.randn(64, 42, 64, 64)

k3x3 = torch.randn(42, 1, 3, 3)
b3x3 = torch.randn(42)

# conv-3x3
v1 = F.conv2d(input, k3x3, b3x3, padding=1, groups=42)
v2 = v1 + input

# re-param conv kernel
weight_idt = torch.zeros(42, 1, 3, 3)
for i in range(42):
    weight_idt[i, 0, 1, 1] = 1.0
r3x3 = k3x3 + weight_idt
print(r3x3.shape)

# compare
v3 = F.conv2d(input, r3x3, b3x3, padding=1, groups=42)
print(torch.sum(v3 - v2))
