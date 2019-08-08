import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

class BaseDetector(object):
    def __init__(self, options):
        if options.gpu:
            try:
                ctx = mx.gpu()
                _ = nd.zeros((1,), ctx=ctx)
            except mx.base.MXNetError:
                print("No GPU available. Use CPU instead.")
                ctx = mx.cpu()
        else:
            ctx = mx.cpu()
        
        print("Creating model...")