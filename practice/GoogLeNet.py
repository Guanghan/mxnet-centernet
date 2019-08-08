from mxnet import gluon, init, nd
from mxnet.gluon import nn

class Inception(nn.Block):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 1st path 
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')

        # 2nd path
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')

        # 3rd path
        self.p3_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c2[1], kernel_size=5, padding=2, activation='relu')

        # 4th path
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')
    
    def forward(self, x):
        # parallel branches
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p3_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1)


def GoogLeNet_network():
    # 1st block
    b1 = nn.Sequential()
    b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    # 2nd block
    b2 = nn.Sequential()
    b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
           nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    # 3rd block
    b3 = nn.Sequential()
    b3.add(Inception(64, (96, 128), (16, 32), 32),
           Inception(128, (128, 192), (32,96), 64),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    # 4th block
    b4 = nn.Sequential()
    b4.add(Inception(192, (96, 208), (16, 48), 64),
           Inception(128, (112, 224), (24,64), 64),
           Inception(128, (128, 256), (24, 64), 64),
           Inception(256, (160, 320), (32,128), 128),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    # 5th block
    b5 = nn.Sequential()
    b5.add(Inception(256, (160, 320), (32,128), 128),
            Inception(384, (192, 284), (48, 128), 128),
            nn.GlobalAvgPool2D()
    )

    net = nn.Sequential()
    net.add(b1, b2, b3, b4, b5, nn.Dense(10))
    return net


def peek_network(net, input_shape=(96, 96)):
    w, h = input_shape
    X = nd.random.uniform(shape=(1,1,w,h))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape: {}'.format(X.shape))
    return


net = GoogLeNet_network()
peek_network(net)