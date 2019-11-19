import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        print('hidden: ', x)
        return self.output(x)



net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
y = net(x)
print(y)

net.hybridize()
y = net(x)
print(y)
