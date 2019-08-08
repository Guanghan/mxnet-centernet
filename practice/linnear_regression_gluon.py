from mxnet import autograd, nd

# generate ground truth function; generate synthetic data from this functionn (plus noise to make it realistic)
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# define the data iterator / loader
from mxnet.gluon import data as gdata
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# test data loading
for X, y in data_iter:
    print(X, y)
    break

# define the network
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))

# initialize parameters
from mxnet import init
net.initialize(init.Normal(sigma = 0.01))

# define loss function
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()

# define the optimization method
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})

# training
num_epoches = 3
for epoch in range(1, num_epoches+1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        # get gradients
        l.backward()
        # update parameters
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

dense = net[0]
print("true_w: {}, w: {}".format(true_w, dense.weight.data().asnumpy()))
print("true_b: {}, b: {}".format(true_b, dense.bias.data().asnumpy()))