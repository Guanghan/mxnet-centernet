

from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    def __init__(self, **kwargs):  # kwargs: keyword arguments
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)
    
    def forward(self, x):
        return self.output(self.hidden(x))

X = nd.random.uniform(shape=(2,20))
net = MLP()
net.initialize()
output = net(X)

print(output)
print(type(net.params))


net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

print(net.params)
print(type(net.params))
print(net[0].params)
print(type(net[0].params))

net[0].params['dense2_weight']
net[0].weight