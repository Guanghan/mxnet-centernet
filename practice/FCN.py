import d2lzh as d2l
from mxnet import gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
import numpy as np
import sys

def trans_conv(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i+h, j:j+w] += X[i,j] * K
    return Y

pretrained_net = model_zoo.vision.resnet18_v2(pretrained = True)
print("Last four layers in feature extraction: \n {} \n".format(pretrained_net.features[-4:]))
print("The output layer: ", pretrained_net.output)

net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

X = nd.random.uniform(shape=(1,3,320,480))
Y = net(X)
print(Y.shape)

num_classes = 21
net.add(
        nn.Conv2D(num_classes, kernel_size=1),
        
        # kernel_size= 2S, padding = S/2, strides = S ->  upsample resolution by S times
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32)
)

def bilinear_kernel(in_channels, output_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, output_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(output_channels), :, :] = filt
    return nd.array(weight)

conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3,3,4)))
