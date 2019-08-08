#%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random


# generate ground truth function; generate synthetic data from this functionn (plus noise to make it realistic)
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(loc = 0, scale = 1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
print("Artificial data sample [0]: ", features[0], labels[0])


def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize = (3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:,1].asnumpy(), labels.asnumpy(), 1)


# Data iterator / loader
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i+batch_size, num_examples)])
        yield features.take(j), labels.take(j)

# test data loading
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print("Loaded data from data_iter function: ", X, y)
    break


# initialize parameters w and b
w = nd.random.normal(scale = 0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()


# define the model
def linreg(X, w, b):
    return nd.dot(X, w) + b

# define the loss function
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# define the optimization method: how to modify the weights given the gradients
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# start training 
lr = 0.03
num_epoches = 3
net = linreg
loss = squared_loss

for epoch in range(num_epoches):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()   # equals to l.sum().backward()
        sgd([w,b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

print('w = {}, true_w = {}'.format(w.asnumpy(), true_w))
print('b = {}, true_b = {}'.format(b.asnumpy(), true_b))