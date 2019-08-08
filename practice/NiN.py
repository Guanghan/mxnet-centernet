from mxnet import gluon, init, nd
from mxnet.gluon import nn

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'),
            nn.Conv2D(num_channels, kernel_size = 1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size = 1, activation='relu')
           )
    return blk

def nin_network():
    net = nn.Sequential()
    net.add(nin_block(96, kernel_size=11, strides=4, padding=8),
            nn.MaxPool2D(pool_size = 3, strides= 2),
            nin_block(256, kernel_size= 5, strides=1, padding=2),
            nn.MaxPool2D(pool_size = 3, strides = 2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2D(pool_size = 3, strides = 2),
            nn.Dropout(0.5),
            nin_block(10, kernel_size=3, strides=1, padding=1),
            nn.GlobalAvgPool2D(),
            nn.Flatten()
    )
    return net

def peek_network(net, input_shape=(224, 224)):
    w, h = input_shape
    X = nd.random.uniform(shape=(1,1,w,h))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape: {}'.format(X.shape))
    return

net = nin_network()
peek_network(net)