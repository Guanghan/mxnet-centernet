from mxnet import gluon, init, nd
from mxnet.gluon import nn

def vgg_block(num_convs, num_channels):
    block = nn.Sequential()

    for _ in range(num_convs):
        layer = nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu')
        block.add(layer)

    layer = nn.MaxPool2D(pool_size=2, strides=2)
    block.add(layer)
    return block

conv_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))

def vgg(conv_arch):
    net = nn.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    
    net.add(nn.Dense(4096, activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), 
            nn.Dropout(0.5),
            nn.Dense(10)
            )
    return net


net = vgg(conv_arch)
print(net)


ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
print(net)