from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import mxnet as mx
import os, sys, time

net = nn.Sequential()
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dense(4096, activation='relu'), 
        nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), 
        nn.Dropout(0.5),
        nn.Dense(10))

X = nd.random.uniform(shape= (1,1,224,224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape: \t', X.shape)


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('~', '.mxnet/datasets/fashion-mnist')):
    root = os.path.expanduser(root)
    transforms = []
    if resize:
        transforms += [gdata.vision.transforms.Resize(resize)]
    transforms += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transforms)

    mnist_train = gdata.vision.FashionMNIST(root = root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root = root, train=False)

    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epoches):
    print('training on: ', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epoches):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            #print('y = ', y)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            #print('y_hat = ', y_hat)
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_accuracy = evaluate_accuracy(test_iter, net, ctx)
        print('epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}, time {:.1f} sec'.format(epoch+1, train_l_sum/n, train_acc_sum / n, test_accuracy, time.time() - start))

def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1)==y).sum()
        n += y.size
    return acc_sum.asscalar() / n


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr, num_epoches, ctx = 0.01, 5, mx.cpu()
net.initialize(force_reinit= True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epoches)