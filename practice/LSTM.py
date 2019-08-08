from mxnet import nd
from mxnet.gluon import rnn

num_inputs, num_hiddens, num_outputs = 128, 256, 128


def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_inputs, num_hiddens)),
                nd.zeros(num_hiddens))

    # params for input gate
    W_xi, W_hi, b_i = _three()
    
    # params for forget gate
    W_xf, W_hf, b_f = _three()
    
    # params for output gate
    W_xo, W_ho, b_o = _three()

    # params for candidate memory cell
    W_xc, W_hc, b_c = _three()

    # params for output layers
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

def init_lstm_state(batch_size, num_hiddens):
    return (nd.zeros(shape=(batch_size, num_hiddens)), nd.zeros(shape=(batch_size, num_hiddens)))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc), b_c)

        C = F*C + I*C_tilda
        H = O * C.tanh()
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return  outputs, (H, C)