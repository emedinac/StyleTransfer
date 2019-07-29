import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import time
import pandas as pd
import numpy as np


class Pipelines(Dataset):
    """Pipeline segmentation dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.size = Image.open(  os.path.join(self.root_dir, self.images['image'].iloc[0]) ).size
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = [Image.open(  os.path.join(self.root_dir, self.images['image'].iloc[idx])  ), 
                Image.open(  os.path.join(self.root_dir, self.images['mask'].iloc[idx])  )] # ONE CLASS
        if self.transform:
            data = self.transform(data)
        return data
    def __sizeof__(self):
        return self.size
        




if __name__ == '__main__':
    '''
    train50.csv:    50%
    test50.csv:     50%
    train01.csv:    1%
    train05.csv:    5%
    train10.csv:    10%
    train25.csv:    25%
    train40.csv:    40%
    '''
    # RUN THIS IN UTILS PATH, REASON:  '../'
    # PipeDataset = Pipelines(csv_file='../train50.csv', root_dir='../Database/Roberto/DATASET3/')
    from transform import *
    import cv2
    PipeDataset = Pipelines(csv_file='../Database/Roberto/DATASET3/test50_c0.csv',
                            root_dir='../Database/Roberto/DATASET3/',
                            transform=transforms.Compose([
                                # CustomResize(256),
                                CustomRandomShift(((0.2,0.2),10)),
                                # CustomRandomHorizontalFlip(0.5),
                                # CustomRandomVerticalFlip(0.5),
                                # CustomRandomRotation([0,90,180,270]),
                                CustomToTensor(),
                                # CustomNormalize(mean=[0.44938242,0.3886864,0.4407186],std=[0.0088543,0.00722883,0.00715833]),
                            ]))
    dataloader = DataLoader(PipeDataset, batch_size=1, shuffle=True, num_workers=32)
    t1 = time.time()
    for i, data in enumerate(dataloader):
        img, mask = data
        cv2.imwrite('image.png',np.uint8(img[0].permute(1,2,0).numpy()*255))
        cv2.imwrite('mask.png',np.uint8(mask[0][0].numpy()*255))
        xxx
        if i==5000:
            break;
    t2 = time.time()
    print('reading time: ', (t2-t1)/i)
    # print('reading time: ', (t2-t1)/PipeDataset.__len__())