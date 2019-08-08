'''
Objects as Points
Mxnet adaptation of the Official Pytorch Implementation

Author: Guanghan Ning
Date: August, 2019
'''

from mxnet import gluon, init, nd
from mxnet.gluon import nn

class convolution(nn.Block):
    def __init__(self, kernel_size, channels_out, channels_in=0, strides=1, with_bn=True, **kwargs):
        super(convolution, self).__init__(**kwargs)
        paddings = (kernel_size - 1)//2  # determine paddings to keep resolution unchanged
        self.conv = nn.Conv2D(channels_out, kernel_size, strides, paddings, in_channels=channels_in, use_bias= not with_bn) # infer input shape if not specified
        self.bn = nn.BatchNorm(in_channels= channels_out) if with_bn else nn.Sequential()
    
    def forward(self, X):
        conv = self.conv(X)
        bn = self.bn(conv)
        return nd.relu(bn)


def test_convolution_shape():
    blk = convolution(kernel_size=3, channels_out=128)
    blk.initialize()
    X = nd.random.uniform(shape=(1, 64, 128, 128))
    Y = blk(X)
    print(Y.shape)


class fully_connected(nn.Block):
    def __init__(self, channels_out, channels_in = 0, with_bn=True, **kwargs):
        super(fully_connected, self).__init__(**kwargs)
        self.with_bn = with_bn
        self.linear = nn.Dense(channels_out, in_units=channels_in) if channels_in else nn.Dense(channels_out)
        if self.with_bn:
            self.bn = nn.BatchNorm(in_channels=channels_out)
    
    def forward(self, X):
        linear = self.linear(X)
        bn = self.bn(linear) if self.with_bn else linear
        return nd.relu(bn)


def test_fully_connected_shape():
    blk = fully_connected(channels_out=128)
    blk.initialize()
    X = nd.random.uniform(shape=(1, 2, 32, 32))
    Y = blk(X)
    print(Y.shape)















def test_all():
    funcs = [test_convolution_shape, test_fully_connected_shape]
    for func in funcs:
        print("Testing routine: {}".format(func.__name__))
        func()


if __name__ == "__main__":
    test_all()
