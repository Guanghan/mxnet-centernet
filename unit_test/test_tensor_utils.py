import sys
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/")

from models.tensor_utils import _gather_feat
import mxnet as mx
from mxnet import nd, gluon, init, autograd

'''
def _gather_feat(feat, ind, mask=None):
    batch_size  = feat.shape[0]

    K = ind.shape[1]
    batch_size = ind.shape[0]
    attri_dim = feat.shape[2]
    output = nd.zeros(shape=(K, batch_size, attri_dim))

    for i in range(batch_size):
        output[:, i, :] = feat[i, ind[i], :]
    return output
'''

def _gather_feat(feat, ind, mask=None):
    # K cannot be 1 for this implementation
    K = ind.shape[1]
    batch_size = ind.shape[0]
    attri_dim = feat.shape[2]

    flatten_ind = ind.flatten()
    for i in range(batch_size):
        if i == 0:
            output = feat[i, ind[i]].expand_dims(2)
        else:
            output = nd.concat(output, feat[i, ind[i]].expand_dims(2), dim=2)

    output = output.swapaxes(dim1 = 1, dim2 = 2)
    return output


def test_gather_feat():
    feat = nd.zeros(shape=(32, 16, 2))  # batch_size = 1, 4x4, attribute 1
    ind = nd.zeros(shape=(32, 2))  # batch_size = 1, top 2

    for i in range(16):
        feat[0, i, 0:2] = i
    print("feat: ", feat)

    ind[0, 0] = 14
    ind[0, 1] = 8
    print("ind: ", ind)

    selected_feat = _gather_feat(feat, ind)
    print("selected_feat", selected_feat)
    return



if __name__ == "__main__":
    test_gather_feat()
