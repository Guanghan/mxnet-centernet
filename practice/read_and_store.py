# save and load nd array
from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x.save', x)

x2 = nd.load('x.save')
print(x2)

y = nd.zeros(4)
nd.save('xy.save', [x,y])
x2, y2 = nd.load('xy.save')
print(x2, y2)

mydict = {'x': x, 'y': y}
nd.save('mydict.save', mydict)
mydict2 = nd.load('mydict.save')
print(mydict2)

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = nd.random.uniform(shape = (2, 20))
Y = net(X)
print(Y)
net.save_params('mlp.params')

net2 = MLP()
net2.load_params('mlp.params')
print(net2)
Y2 = net2(X)
print(Y2 == Y)

import mxnet as mx
a = nd.array([1,2,3], ctx = mx.cpu(1))
print(a)