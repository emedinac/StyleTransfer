import cv2
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,64,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,32,3,1,1))
        self.fc = nn.Linear(32*32,32*32)

    def forward(self,x,masks,style=False):
        color_code_number = 9
        xb,xc,xh,xw = x.size()
        x = x.view(xc,-1)
        feature_sub_mean = x.clone()
        for i in range(color_code_number):
            mask = masks[i].clone().squeeze(0)
            mask = cv2.resize(mask.numpy(),(xw,xh),interpolation=cv2.INTER_NEAREST)
            mask = torch.FloatTensor(mask)
            mask = mask.long()
            if(torch.sum(mask) >= 10):
                mask = mask.view(-1)
                fgmask = (mask==1).nonzero().squeeze(1)
                fgmask = fgmask.cuda()
                selectFeature = torch.index_select(x,1,fgmask) # 32x96
                # subtract mean
                f_mean = torch.mean(selectFeature,1)
                f_mean = f_mean.unsqueeze(1).expand_as(selectFeature)
                selectFeature = selectFeature - f_mean
                feature_sub_mean.index_copy_(1,fgmask,selectFeature)

        feature = self.convs(feature_sub_mean.view(xb,xc,xh,xw))
        # 32x16x16
        b,c,h,w = feature.size()
        transMatrices = {}
        feature = feature.view(c,-1)

        for i in range(color_code_number):
            mask = masks[i].clone().squeeze(0)
            mask = cv2.resize(mask.numpy(),(w,h),interpolation=cv2.INTER_NEAREST)
            mask = torch.FloatTensor(mask)
            mask = mask.long()
            if(torch.sum(mask) >= 10):
                mask = mask.view(-1)
                fgmask = (mask==1).nonzero().squeeze(1)
                fgmask = fgmask.cuda()
                selectFeature = torch.index_select(feature,1,fgmask) # 32x96
                tc,tN = selectFeature.size()

                covMatrix = torch.mm(selectFeature,selectFeature.transpose(0,1)).div(tN)
                transmatrix = self.fc(covMatrix.view(-1))
                transMatrices[i] = transmatrix
        return transMatrices,feature_sub_mean

class MulLayer(nn.Module):
    def __init__(self,layer):
        super(MulLayer,self).__init__()
        self.snet = CNN()
        self.cnet = CNN()
        self.compress = nn.Conv2d(256,32,1,1,0)
        self.unzip = nn.Conv2d(32,256,1,1,0)

    def forward(self,cF,sF,cmasks,smasks):

        sb,sc,sh,sw = sF.size()

        sMatrices,sF_sub_mean = self.snet(sF,smasks,style=True)
        cMatrices,cF_sub_mean = self.cnet(cF,cmasks,style=False)

        compress_content = self.compress(cF_sub_mean.view(cF.size()))
        cb,cc,ch,cw = compress_content.size()
        compress_content = compress_content.view(cc,-1)
        transfeature = compress_content.clone()
        color_code_number = 9
        finalSMean = torch.zeros(cF.size()).cuda(0)
        finalSMean = finalSMean.view(256,-1)
        for i in range(color_code_number):
            cmask = cmasks[i].clone().squeeze(0)
            smask = smasks[i].clone().squeeze(0)

            cmask = cv2.resize(cmask.numpy(),(cw,ch),interpolation=cv2.INTER_NEAREST)
            cmask = torch.FloatTensor(cmask)
            cmask = cmask.long()
            smask = cv2.resize(smask.numpy(),(sw,sh),interpolation=cv2.INTER_NEAREST)
            smask = torch.FloatTensor(smask)
            smask = smask.long()
            if(torch.sum(cmask) >= 10 and torch.sum(smask) >= 10
               and (i in sMatrices) and (i in cMatrices)):
                cmask = cmask.view(-1)
                fgcmask = (cmask==1).nonzero().squeeze(1)
                fgcmask = fgcmask.cuda()

                smask = smask.view(-1)
                fgsmask = (smask==1).nonzero().squeeze(1)
                fgsmask = fgsmask.cuda()

                sFF = sF.view(sc,-1)
                sFF_select = torch.index_select(sFF,1,fgsmask)
                sMean = torch.mean(sFF_select,dim=1,keepdim=True)
                sMean = sMean.view(1,sc,1,1)
                sMean = sMean.expand_as(cF)

                sMatrix = sMatrices[i]
                cMatrix = cMatrices[i]

                sMatrix = sMatrix.view(32,32)
                cMatrix = cMatrix.view(32,32)

                transmatrix = torch.mm(sMatrix,cMatrix) # (C*C)

                compress_content_select = torch.index_select(compress_content,1,fgcmask)

                transfeatureFG = torch.mm(transmatrix,compress_content_select)
                transfeature.index_copy_(1,fgcmask,transfeatureFG)

                sMean = sMean.contiguous()
                sMean_select = torch.index_select(sMean.view(sc,-1),1,fgcmask)
                finalSMean.index_copy_(1,fgcmask,sMean_select)
        out = self.unzip(transfeature.view(cb,cc,ch,cw))
        return out + finalSMean.view(out.size())
