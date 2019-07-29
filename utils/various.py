import os

import torch, cv2
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

def get_iou(pred, gt, n_classes=21):
    pred = torch.max(pred, 1)[1]
    total_miou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        miou = (sum(iou) / len(iou))
        total_miou += miou

    return total_miou

# Modify this initializer
def init_weight(self, init_type, backbone_pretrained='', model_pretrained=''):
    if not model_pretrained:
        print('Initialazing weights in all the model')
        init_type = init_type.lower()
        if backbone_pretrained:
            backbone_pretrained = True
            print('Backbone weights were loaded')
        for n, m in self.named_modules():
            if backbone_pretrained and 'backbone' in n:
                continue;
            if isinstance(m, nn.Conv2d):
                if 'he' in init_type:
                    if init_type=='gauss': init.kaiming_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    if init_type=='uniform': init.kaiming_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                elif 'xavier' in init_type:
                    if init_type=='gauss': init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    if init_type=='uniform': init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                elif 'uniform' in init_type:
                    v = init_type[init_type.find(',')+1:].split(',')
                    init.uniform_(m.weight, a=float(v[0]), b=float(v[1]))
                elif 'gauss' in init_type:
                    v = init_type[init_type.find(',')+1:].split(',')
                    init.normal_(m.weight, mean=float(v[0]), std=float(v[1]))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if n == 'ClassConv':
                init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
    else:
        pretrain_dict = torch.load(model_pretrained)['state_dict']
        state_dict = self.state_dict()
        model_dict = {}
        keywords = [s for s in state_dict.keys() if 'weight' in s or 'bias' in s]
        n=0;
        for k, v in pretrain_dict.items():
            if 'fc' in k:
                break;
            if 'weight' in k or 'bias' in k:
                if 'pointwise' in k:
                    # print('1')
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if v.shape!=state_dict[keywords[n]].shape:
                    print(keywords[n])
                    n += 1
                    continue;
                model_dict[keywords[n]] = v
                n += 1
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)