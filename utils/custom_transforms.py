from PIL import Image
import random
import math
import numpy as np
import torch

from Networks.StyleNet import StyleAugmentation
import torch.nn.functional as F

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class Stylization(object):
    def __init__(self, layer='r31', alpha=0.05, prob=0.1, pseudo1=True, Noise=False, std=1., mean=0.):
        self.net = StyleAugmentation(layer, alpha, prob, pseudo1, Noise, std, mean).cuda() # CUDA is available
        for name, child in self.net.named_children():
            for param in child.parameters():
                param.requires_grad = False
        
    def __call__(self, x):
        with torch.no_grad():
            return self.net(x[None,:].cuda())[0].cpu()
        # with torch.no_grad():
        #     x = self.net(x[None,:,:-1,:-1].cuda()) 
        # return F.pad(x, (0,1,0,1))[0].cpu()

if __name__ == "__main__":
    import os
    os.environ['QT_QPA_PLATFORM']='offscreen'
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    from torchvision.transforms import *

    # Testing the Stylization algorithm.
    transformations_set_aug = [ transforms.Resize(229, interpolation=2),
                                
                                ]
    transformations_set_aug.extend([ transforms.ToTensor(),
                                        Stylization(layer='r31', # (0.25 , 0.5, False, False, 3., 0.) 
                                                    alpha=0.25,
                                                    prob=0.5,
                                                    pseudo1=False,
                                                    Noise=False,
                                                    std=1.0,
                                                    mean=0.0,),
                                        transforms.ToPILImage(),   ]) # Optimize this section

    transformations_set_aug.extend([ transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.95,1.05), shear=10, resample=False, fillcolor=0),
                                        transforms.RandomRotation(10, resample=False, expand=False, center=None),
                                        transforms.ToTensor(),
                                        RandomErasing(probability=0.5, sh=0.4, r1=0.3),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])

    trainset = torchvision.datasets.STL10(root='./Database/', 
                                            split='train',  
                                            transform=transforms.Compose(transformations_set_aug), 
                                            target_transform=None, 
                                            download=True)
    batch = 256
    isize = 16
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=False, num_workers=0)
    print("running...")
    # Apply each of the above transforms on sample.
    fig, ax = plt.subplots(isize, batch//isize, figsize=(20,20))
    for it, (imgs, targets) in enumerate(trainloader):
        imgs = (imgs - imgs.min()) / (imgs.max()-imgs.min())
        imgs = imgs.transpose(1,2).transpose(2,3).numpy();
        print(it)

        for j in range(imgs.shape[0]):        
            ax[j//isize, j%isize].imshow(imgs[j])
            ax[j//isize, j%isize].set_axis_off()
        break;
    plt.savefig("test_augmentation_normal.png", bbox_inches = 'tight', pad_inches = 0, dpi=500)
