from mxnet import nd
from mxnet.gluon import loss as gloss

from fashion_mnist import train_iter, test_iter, get_fashion_mnist_labels, show_fashion_mnist
from softmax_regression import train_ch3

batch_size = 256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


def relu(X):
    return nd.maximum(X, 0)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2


loss = gloss.SoftmaxCrossEntropyLoss()

num_epoches, lr = 5, 0.5
train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, params, lr, None)