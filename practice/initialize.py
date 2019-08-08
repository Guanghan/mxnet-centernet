from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init: ', name, data.shape)


# Delayed initialization
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
net.initialize(init = MyInit())

X = nd.random.uniform(shape = (2,20))
Y = net(X)

# immediate initialization
# (1) re-initialize models that have been initialized before
net.initialize(init= MyInit(), force_reinit=True)
X = nd.random.uniform(shape = (2,40))
Y = net(X)

# (2) explicitly give the input dimension, so that the system do not need additional info to initialize
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))
net.initialize(init = MyInit())

