import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms, utils
from PIL import Image

class CustomResize(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        sample[0] = TF.resize(sample[0], size=self.output_size, interpolation=Image.BILINEAR)  
        sample[1] = TF.resize(sample[1], size=self.output_size, interpolation=Image.NEAREST)
        return sample

class CustomRandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.crop = transforms.RandomCrop(size=self.output_size)

    def __call__(self, sample):
        for c, image in enumerate(sample):
            sample[c] = self.crop(image)
        return sample

class CustomRandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        prob = np.random.random()
        for c, image in enumerate(sample):
            if prob<self.prob:
                sample[c] = TF.hflip(image)
        return sample

class CustomRandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        prob = np.random.random()
        for c, image in enumerate(sample):
            if prob<self.prob:
                sample[c] = TF.vflip(image)
        return sample

class CustomRandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
    def __call__(self, sample):
        degree = np.random.choice(self.degrees)
        for c, image in enumerate(sample):
            sample[c] = TF.rotate(image, degree)
        return sample

class CustomRandomShift(object):
    def __init__(self, config):
        rate = config[0]
        self.shear= config[1]
        assert isinstance(rate, (int, list))
        if isinstance(rate, int):
            self.rate = [rate, rate]
        else:
            self.rate = rate
    def __call__(self, sample):
        ratex = np.random.uniform(-self.rate[0],self.rate[0])
        ratey = np.random.uniform(-self.rate[1],self.rate[1])
        X,Y = sample[0].size
        dx = int(ratex*X)
        dy = int(ratey*Y)
        shear = np.random.uniform(-self.shear,self.shear)
        for c, image in enumerate(sample):
            sample[c] = TF.affine(image,0,(dx,dy), scale=1.0, shear=shear)
        return sample

class CustomNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        sample[0] = TF.normalize(sample[0], self.mean, self.std)
        return sample


class CustomToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, maxval=1.):
        sample[0] = TF.to_tensor(sample[0])
        sample[1] = TF.to_tensor(sample[1])*maxval
        return sample

