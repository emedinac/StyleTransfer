import os, cv2
import pickle
import torch
import torch.nn as nn
import torchvision
from libs.Loader import Dataset

import numpy as np

from libs.Matrix import CNN
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4

# Modified from Original: https://github.com/sunshineatnoon/LinearStyleTransfer
class MulLayer(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = CNN(layer,matrixSize)
        self.cnet = CNN(layer,matrixSize)
        self.matrixSize = matrixSize

        if(layer == 'r41'):
            self.compress = nn.Conv2d(512,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,512,1,1,0)
        elif(layer == 'r31'):
            self.compress = nn.Conv2d(256,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,256,1,1,0)
        elif(layer == 'r21'):
            self.compress = nn.Conv2d(128,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,128,1,1,0)
        elif(layer == 'r11'):
            self.compress = nn.Conv2d(64,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,64,1,1,0)
        self.transmatrix = None

    def forward(self,cF,sF,means,alpha=1.0):
        cFBK = cF.clone()
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sMean = means.unsqueeze(2).unsqueeze(3)
        sMeanC = sMean.expand_as(cF)

        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        cMatrix = self.cnet(cF)
        sMatrix = self.snet(cF)*alpha + sF*(1-alpha) # only a 1024 vector

        sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
        cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
        transmatrix = torch.bmm(sMatrix,cMatrix)
        transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
        out = self.unzip(transfeature.view(b,c,h,w))
        out = out + cMean*(alpha) + sMeanC*(1-alpha)
        return out, transmatrix


class StyleAugmentation(nn.Module):
    def __init__(self,layer='r31',std=1.,mean=0.):
        super(StyleAugmentation,self).__init__()
        # Open - Load
        with open('features.p', 'rb') as handle:
            self.features, self.means = pickle.load(handle)
        self.size = len(self.features)
        print("number of style available: ", self.size)
        self.matrix = MulLayer('r31')
        self.vgg = encoder3()
        self.dec = decoder3()
        self.vgg.load_state_dict(torch.load('models/vgg_'+layer+'.pth'))
        self.dec.load_state_dict(torch.load('models/dec_'+layer+'.pth'))
        self.matrix.load_state_dict(torch.load('models/'+layer+'.pth'))
        self.dist = torch.distributions.normal.Normal(torch.tensor([mean]), torch.tensor([std]))
    def forward(self, x, pseudo1=True):
        b = x.size(0)
        if pseudo1: # get 1 and add noise to each sample
            idx = np.random.randint(0,self.size,1)
            idx = 0
            # sF = torch.cuda.FloatTensor(self.features[idx])+self.dist.sample((b,1024)).squeeze(2).cuda()
            sF = torch.cuda.FloatTensor(self.features[idx]).unsqueeze(0).repeat([b,1])
            Fm = torch.cuda.FloatTensor(self.means[idx]).repeat([b,1])
            print(self.dist.sample((b,1024)).shape, torch.cuda.FloatTensor(self.features[idx]).shape, sF.shape, Fm.shape)
        else:
            idx = np.random.randint(0,self.size,b)
            sF = torch.cuda.FloatTensor(self.features[idx])
            Fm = torch.cuda.FloatTensor(self.means[idx])
        cF = self.vgg(x)
        feature,transmatrix = self.matrix(cF,sF,Fm)
        transfer = self.dec(feature)
        return transfer.clamp(0,1)


if __name__ == "__main__":
    batch_size = 8
    content_dataset = Dataset('Database/COCO/2017/train2017/',256,256,test=True)
    content_loader = torch.utils.data.DataLoader(dataset     = content_dataset,
                                                  batch_size  = batch_size,
                                                  shuffle     = False,
                                                  num_workers = 1,
                                                  drop_last   = True)

    Stylenet = StyleAugmentation().cuda()
    for it, (content,_) in enumerate(content_loader):
        styled = Stylenet(content.cuda())

        # vutils.save_image(styled[0].data,'Style.png',normalize=True,scale_each=True,nrow=1)
        for n in range(styled.shape[0]):
            Image = np.uint8(content.permute(0,2,3,1)[n].cpu().detach().numpy()*255)
            Style0 = np.uint8(styled.permute(0,2,3,1)[n].cpu().data.numpy()*255)
            cv2.imwrite(str(n).zfill(3)+'Style.png',cv2.cvtColor( Style0 ,cv2.COLOR_BGR2RGB))
            cv2.imwrite(str(n).zfill(3)+'Image.png',cv2.cvtColor( Image ,cv2.COLOR_BGR2RGB))
        break;