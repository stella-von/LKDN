import torch

dict = torch.load('experiments/pretrained_models/LKDN/LKDN_x4.pth')  # load the pth file

for key in list(dict.keys()):  # print the keys
    print(key)

print('\n-------------------\n')

for key in list(dict['params_ema'].keys()):
    print(key, end=' | ')
    print(dict['params_ema'][key].shape)
