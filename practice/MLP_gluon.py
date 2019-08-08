from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
from fashion_mnist import train_iter, test_iter, get_fashion_mnist_labels, show_fashion_mnist
from softmax_regression import train_ch3

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
net.initialize(init.Normal(sigma = 0.01))

batch_size = 256
num_epoches = 5

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

print("Start training MLP with gluon:")
train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, None, None, trainer)