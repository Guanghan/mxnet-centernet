#%matplotlib inline
import d2lzh as d2l 
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils

import sys
import time

d2l.set_figsize()
img = image.imread('../../d2l-zh/img/cat1.jpg')
d2l.plt.imshow(img.asnumpy())

# some common augmentation methods
gdata.vision.transforms.RandomFlipLeftRight()
gdata.vision.transforms.RandomFlipTopBottom()
crop_aug = gdata.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))

gdata.vision.transforms.RandomBrightness(0.5)
gdata.vision.transforms.RandomHue(0.5)
color_aug = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

# use multiple augmentation methods
aug_method_list = []
aug_method_list.append(color_aug)
aug_method_list.append(crop_aug)
aug_method_list.append(gdata.vision.transforms.ToTensor())
augs = gdata.vision.transforms.Compose(aug_method_list)

# create data loader
feed = gdata.vision.CIFAR10().transform_first(augs)
data_loader = gdata.DataLoader(feed, batch_size=64, shuffle=True, num_workers=10)
