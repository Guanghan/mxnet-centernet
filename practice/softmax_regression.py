from mxnet import autograd, nd
from fashion_mnist import train_iter, test_iter, get_fashion_mnist_labels, show_fashion_mnist
from linear_regression import sgd

batch_size = 256
num_inputs = 784
num_outputs = 10

W = nd.random.normal(loc = 0, scale = 0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()

X = nd.array([[1,2,3], [4,5,6]])
print(X.sum(axis=0, keepdims=True))
print(X.sum(axis=1, keepdims=True))

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis = 1, keepdims=True)
    return X_exp / partition   # applies broadcast mechanism here

# check the effect of softmax
X = nd.random.normal(shape = (2,5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(axis=1))

def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


# check the effect of function nd.pick
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0,2], dtype='int32')
nd.pick(y_hat, y)

def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

# test accuracy
print(accuracy(y_hat, y))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n

# test evaluate_accuracy
print(evaluate_accuracy(test_iter, net))

num_epoches, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epoches):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)

            train_l_sum += l.asscalar()
            y = y.astype('float32')
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch {}, loss {:.2f}, train acc {:.3f}, test acc {:.3f}'.format(epoch+1, train_l_sum / n, train_acc_sum / n, test_acc))

# start training!
train_ch3(net, train_iter, test_iter, cross_entropy, num_epoches, batch_size, [W,b], lr)


# start inference!
for X, y in test_iter:
    #print(X)
    break
true_labels = get_fashion_mnist_labels(y.asnumpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true,pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])