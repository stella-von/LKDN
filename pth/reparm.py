import torch
import torch.nn.functional as F

dict = torch.load('LKDN-S_del_x4.pth')

blk_name = list(dict.keys())

for i in range(len(blk_name)):
    if blk_name[i].endswith('r.pw.weight'):
        weight = dict[blk_name[i]]
        weight_idt = torch.zeros(42, 42, 1, 1)
        for j in range(42):
            weight_idt[j, j, 0, 0] = 1.0
        rep_weight = weight + weight_idt
        dict[blk_name[i]] = rep_weight
    elif blk_name[i].endswith('r.dw.weight'):
        weight, bias = dict[blk_name[i]], dict[blk_name[i + 1]]
        weight1x1, bias1x1 = dict[blk_name[i + 2]], dict[blk_name[i + 3]]
        weight_idt = torch.zeros(42, 1, 3, 3)
        for j in range(42):
            weight_idt[j, 0, 1, 1] = 1.0
        weight1x1 = F.pad(weight1x1, (1, 1, 1, 1), 'constant', 0.0)
        rep_weight = weight + weight_idt + weight1x1
        rep_bias = bias + bias1x1
        dict[blk_name[i]] = rep_weight
        dict[blk_name[i + 1]] = rep_bias
        del dict[blk_name[i + 2]]
        del dict[blk_name[i + 3]]
    elif blk_name[i] == 'upsampler.conv1.weight':
        weight1, bias1 = dict[blk_name[i]], dict[blk_name[i + 1]]
        weight3, bias3 = dict[blk_name[i + 2]], dict[blk_name[i + 3]]
        print(weight1.size(), weight3.size())
        weight1 = F.pad(weight1, (1, 1, 1, 1), 'constant', 0.0)
        weight1x1, bias1x1 = dict[blk_name[i + 4]], dict[blk_name[i + 5]]
        weight3x3, bias3x3 = dict[blk_name[i + 6]], dict[blk_name[i + 7]]
        rep_weight = F.conv2d(weight3x3, weight1x1.permute(1, 0, 2, 3))
        rep_bias = torch.ones(1, 84, 3, 3) * bias1x1.view(1, -1, 1, 1)
        rep_bias = F.conv2d(rep_bias, weight3x3).view(-1, ) + bias3x3

        rep_weight = rep_weight + weight1 + weight3
        rep_bias = rep_bias + bias1 + bias3
        dict[blk_name[i + 2]] = rep_weight
        dict[blk_name[i + 3]] = rep_bias
        dict['upsampler.0.weight'] = rep_weight
        dict['upsampler.0.bias'] = rep_bias
        del dict[blk_name[i]]
        del dict[blk_name[i + 1]]
        del dict[blk_name[i + 2]]
        del dict[blk_name[i + 3]]
        del dict[blk_name[i + 4]]
        del dict[blk_name[i + 5]]
        del dict[blk_name[i + 6]]
        del dict[blk_name[i + 7]]

torch.save(dict, 'LKDN-S_del_rep_x4.pth')

dict = torch.load('LKDN-S_del_rep_x4.pth')

for key in list(dict.keys()):
    print(key, end=' ')
    print(dict[key].shape)
