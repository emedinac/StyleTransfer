import torch, torchvision
import torch.nn as nn
from torch.nn import init
from .Xception import *
from .InceptionV4 import *

def init_weights(m):
    global init_net
    inet = init_net.split(',')[0]
    dist = init_net.split(',')[1]
    if inet=='xavier':
        if dist=='uniform':
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()
        elif dist=='gauss':
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()
    if inet=='xavier':
        if dist=='uniform':
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()
        elif dist=='gauss':
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_ (m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()

def ChooseNet(name, pretrained=None):
    global init_net
    init_net = pretrained.init # conf.init, pretrained is just for the env.
    if pretrained: 
        pretrained = False
    else: 
        pretrained = pretrained.pretrained;

    if name=="InceptionV3": # Only ImageNet input
        net = torchvision.models.inception_v3(pretrained=pretrained)
        net.fc = nn.Linear(2048,10,bias=True)
    elif name=="InceptionV4": # Only ImageNet input
        net = inceptionv4(pretrained=pretrained)
        net.last_linear = nn.Linear(1536, 10)
    elif name=="VGG16": # Only ImageNet input
        net = torchvision.models.vgg16_bn(pretrained=pretrained)
        net.classifier._modules['0'] = nn.Linear(8192, 4096) 
        net.classifier._modules['6'] = nn.Linear(4096, 10)
    elif name=="Resnet18":
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Linear(512,10,bias=True)
    elif name=="Resnet50":
        net = torchvision.models.resnet50(pretrained=pretrained)
        net.fc = nn.Linear(2048,10,bias=True)
    elif name=="Resnet101":
        net = torchvision.models.resnet101(pretrained=pretrained)
        net.fc = nn.Linear(2048,10,bias=True)
    elif name=="Squeeze11":
        net = torchvision.models.squeezenet1_1(pretrained=pretrained)
        net.num_classes=10
        net.classifier._modules['1'] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
        net.classifier._modules['3'] = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
    elif name=="Xception":
        net = xception(pretrained=pretrained)
        net.fc = nn.Linear(2048, 10, bias=True)
    if not pretrained:
        print(name, " , ", init_net )
        net.apply(init_weights)
    return net