from mxnet import nd
from mxnet.gluon import nn

def comp_conv2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape((1,1) + X.shape)   # add two more dimensions: batch, n_channels
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8,8))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

conv2d = nn.Conv2D(1, kernel_size=(5,3), padding=(2, 1))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

