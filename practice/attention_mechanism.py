import math
from mxnet import nd
from mxnet.gluon import nn

def masked_softmax(X, valid_length):
    if valid_length is None:
        return X.softmax()
    else:
        shape = X.shape
        if valid_length.ndim == 1:
            valid_length = valid_length.repeat(shape[1], axis=0)
        else:
            valid_length = valid_length.reshape((-1,))
        
        X = nd.SequenceMask(X.reshape((-1, shape[-1])), valid_length, True, axis=1, value=-1e6)
        return X.softmax().reshape(shape)


def test_masked_softmax():
    X = nd.random.uniform(shape=(2,2,4))
    valid_length = nd.array([2,3])
    Y = masked_softmax(X, valid_length)
    print("input:", X)
    print("output", Y)


class DotProductAttention(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        print("d.shape", d)
        print("query.shape", query.shape)   # (2,1,2)
        print("key.shape", key.shape)      # (2,10,2) -->  (2,2,10)
        print("key.shape.transpose", key.transpose().shape)      # (2,10,2) -->  (2,2,10)
        scores = nd.batch_dot(query, key, transpose_b=True) / math.sqrt(d)   # transpose the second input
        print("scores.shape", scores.shape)   #(2,1,10)

        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        print("attention weights shape", attention_weights.shape)
        return nd.batch_dot(attention_weights, value)

def test_DotProductAttention():
    atten = DotProductAttention(dropout=0.5)
    atten.initialize()
    keys = nd.ones((2,10,2))
    values = nd.arange(40).reshape((1,10,4)).repeat(2,axis=0)
    output = atten(nd.ones((2,1,2,)), keys, values, nd.array([2,6]))

    print("values.shape = ", values.shape)
    print("output.shape = ", output.shape)


class MLPAttention(nn.Block):
    def __init__(self, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        self.W_k = nn.Dense(units, activation='tanh', use_bias=False, flatten=False)
        self.W_q = nn.Dense(units, activation='tanh', use_bias=False, flatten=False)
        self.v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, valid_length):
        query, key = self.W_k(query), self.W_q(key)
        features = query.expand_dims(axis=2) + key.expand_dims(axis=1)
        scores = self.v(features).squeeze(axis=-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return nd.batch_dot(attention_weights, value)

atten = MLPAttention(units=8, dropout=0.1)
atten.initialize()

keys = nd.ones((2,10,2))
values = nd.arange(40).reshape((1,10,4)).repeat(2,axis=0)
X = nd.ones((2,1,2))
Y = atten(X, keys, values, nd.array([2,6]))
print(Y)

