import pickle
import torch
import torch.nn as nn
import numpy as np

from .libs.Matrix import CNN

import torch.backends.cudnn as cudnn
from .libs.models import encoder3,encoder4
from .libs.models import decoder3,decoder4

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

    def forward(self,cF,sF,means,alpha=0.0): # is the original image (theorically)
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
    def __init__(self,layer='r31', alpha=0.5, prob=0.25, pseudo1=True, Noise=True, std=1.,mean=0.):
        super(StyleAugmentation,self).__init__()
        self.alpha = alpha
        self.prob = prob
        self.pseudo1 = pseudo1
        self.Noise = Noise
        self.std = std
        self.mean = mean
        # Open - Load
        with open('features.p', 'rb') as handle:
            self.features, self.means = pickle.load(handle)
        self.size = len(self.features)
        print("number of style available: ", self.size)
        self.matrix = MulLayer('r31')
        self.vgg = encoder3()
        self.dec = decoder3()
        self.vgg.load_state_dict(torch.load('Networks/models/vgg_'+layer+'.pth'))
        self.dec.load_state_dict(torch.load('Networks/models/dec_'+layer+'.pth'))
        self.matrix.load_state_dict(torch.load('Networks/models/'+layer+'.pth'))
        self.dist = torch.distributions.normal.Normal(torch.tensor([self.mean]), torch.tensor([self.std]))
    def forward(self, x):
        b = x.size(0)
        new_batch = int(b*self.prob)
        if new_batch<1: new_batch = 1
        self.index = torch.unique(torch.randint(b,(new_batch,)))
        temp_batch = len(self.index)
        xtemp = x[self.index]
        if self.pseudo1: # Get 1 and add noise to each sample
            idx = np.random.randint(0,self.size,1)
            Fm = torch.FloatTensor(self.means[idx]).repeat([temp_batch,1]).cuda()
            sF = torch.FloatTensor(self.features[idx]).repeat([temp_batch,1]).cuda()
            if self.Noise: sF += self.dist.sample((temp_batch,1024)).squeeze(2).cuda()
        else: # Get multiple Styles
            idx = np.random.randint(0,self.size,temp_batch)
            Fm = torch.FloatTensor(self.means[idx]).cuda()
            sF = torch.FloatTensor(self.features[idx]).cuda()
            if self.Noise: sF += self.dist.sample((temp_batch,1024)).squeeze(2).cuda()
        # sF = torch.cuda.FloatTensor(self.features[idx]).unsqueeze(0).repeat([temp_batch,1])
        cF = self.vgg(xtemp)
        feature,transmatrix = self.matrix(cF,sF,Fm,self.alpha)
        transfer = self.dec(feature)
        x[self.index] = transfer.clamp(0,1)
        return x


if __name__ == "__main__":
    import os, cv2
    from .libs.Loader import Dataset
    # import torchvision
    # import torchvision.utils as vutils
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