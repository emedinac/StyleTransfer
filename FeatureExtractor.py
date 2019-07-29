from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 8000000000
import torch
import torchvision
import torch.nn as nn
from libs.Loader import Dataset
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4

os.environ["CUDA_VISIBLE_DEVICES"]="0" # USED ONLY IF OTHER GPUS ARE BEING USED
if True:
    style_dataset = Dataset('Database/WikiArt/train/',256,256,test=True)
    style_loader_ = torch.utils.data.DataLoader(dataset     = style_dataset,
                                                batch_size  = 128,
                                                shuffle     = False,
                                                num_workers = 4,
                                                drop_last   = True)
    style_loader = iter(style_loader_)
    # styleV = torch.Tensor(64,3,224,224).cuda()

    matrix = MulLayer('r31')
    vgg = encoder3()
    vgg.load_state_dict(torch.load('models/vgg_r31.pth'))
    matrix.load_state_dict(torch.load('models/r31.pth'))
    vgg.cuda(); 
    matrix.cuda()
    features = []
    means = []
    with torch.no_grad():
        for iteration, (styleV, t) in enumerate(style_loader_):
            sF = vgg(styleV.cuda())
            sb,sc,sh,sw = sF.size()
            sFF = sF.view(sb,sc,-1)
            sMean = torch.mean(sFF,dim=2,keepdim=True)
            sMean = sMean.unsqueeze(3)
            sMeanS = sMean.expand_as(sF)
            sF = sF - sMeanS
            sF = matrix.snet(sF)
            

            features.extend(sF.cpu().numpy().tolist())
            means.extend(sMean[:,:,0,0].cpu().numpy().tolist())
            print(100*iteration/style_loader_.__len__())
    features = np.array(features)
    means = np.array(means)
    with open('features.p', 'wb') as handle:
        pickle.dump([features, means], handle, protocol=pickle.HIGHEST_PROTOCOL)

if False: # For analysis
    # Open - Load - Analysis
    with open('features.p', 'rb') as handle:
        features = pickle.load(handle)[0]
    print(features.shape)

    ########################
    ########################
    # This visulization is performed to understand the feature-level information after stylization.
    ########################
    ########################
    # T-sne takes a long time to finish.

    embedding0 = PCA(n_components=50)
    z_o = embedding0.fit_transform(features)
    print(z_o.shape)
    # embedding1 = TSNE(n_components=2, n_iter=500).fit_transform(features)
    # z1 = embedding1.fit_transform(pca_result_50[rndperm[:n_sne]])
    # print(z1.shape)
    z2 = TSNE(n_components=2, n_iter=500, verbose=1).fit_transform(z_o)
    print(z2.shape)
    z3 = TSNE(n_components=1, n_iter=500, verbose=1).fit_transform(z_o)
    print(z3.shape)
    with open('features_50_2D.p', 'wb') as handle:
        pickle.dump(z2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('features_50_1D.p', 'wb') as handle:
        pickle.dump(z3, handle, protocol=pickle.HIGHEST_PROTOCOL)

    embedding0 = PCA(n_components=100)
    z_o = embedding0.fit_transform(features)
    print(z_o.shape)
    z2 = TSNE(n_components=2, n_iter=1000, verbose=1).fit_transform(z_o)
    print(z2.shape)
    z3 = TSNE(n_components=1, n_iter=1000, verbose=1).fit_transform(z_o)
    print(z3.shape)
    with open('features_100_2D.p', 'wb') as handle:
        pickle.dump(z2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('features_100_1D.p', 'wb') as handle:
        pickle.dump(z3, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open('features_100_1D.p', 'rb') as handle:
        f1 = pickle.load(handle)
    print(f1.shape)
    # import os
    # os.environ['QT_QPA_PLATFORM']='offscreen'
    # import matplotlib.pyplot as plt
    # plt.hist(f1, bins=1000, color='r')
    # plt.show()


    # # plt.scatter(z1[:,0], z1[:,1], color='r')
    # plt.scatter(z2[:,0], z2[:,1], color='g')
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.show()