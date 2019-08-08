from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    
    def forward(self, x):
        return x - x.mean()

# instantiate the class, get an instance
layer = CenteredLayer()
layer(nd.array([1,2,3,4,5]))


net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())

net.initialize()
x= nd.random.uniform(shape=(4,8))
y = net(x)
print(y.shape)
print(y.mean().asscalar())

params = gluon.ParameterDict()
params.get('param2', shape=(2,3))
params.get('param3', shape=(3,4))
print(params)

class MyDense(nn.Block):
    def __init__(self, units_out, units_in, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(units_in, units_out))
        self.bias = self.params.get('bias', shape=(units_out, ))
    
    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)

dense = MyDense(3, 5)
print(dense.params)

dense.initialize()
output = dense(nd.random.uniform(shape=(2,5)))
print(output)


net = nn.Sequential()
net.add(MyDense(8, units_in=64))
net.add(MyDense(1, units_in=8))
net.initialize()
x = nd.random.uniform(shape=(2, 64))
y = net(x)
print(y)
print(net.params)
print(net[0].params)
print(net[1].params)