"""
Author: Guanghan Ning
Date:   August, 2019

Adapted from the official Pytorch implementation:
https://github.com/xingyizhou/CenterNet

Homage to [PyTorch to Mxnet cheetsheet]:
https://gist.github.com/zhanghang1989/3d646f71d60c17048cf8ad582393ac6c
"""

from __future__ import absolute_import
import mxnet
from mxnet import gluon, init, nd
from mxnet.gluon import nn

def _sigmoid(x):
  y = nd.clip(data=x.sigmoid(), a_min=1e-4, a_max=1-1e-4)
  return y


def _gather_feat(feat, ind, mask=None):
    # K cannot be 1 for this implementation
    K = ind.shape[1]
    batch_size = ind.shape[0]
    attri_dim = feat.shape[2]

    flatten_ind = ind.flatten()
    for i in range(batch_size):
        if i == 0:
            output = feat[i, ind[i]].expand_dims(2)   # similar to nd.pick
        else:
            output = nd.concat(output, feat[i, ind[i]].expand_dims(2), dim=2)
    output = output.swapaxes(dim1 = 1, dim2 = 2)
    return output

def _tranpose_and_gather_feat(feat, ind):
    feat = nd.transpose(feat, axes=(0, 2, 3, 1))
    feat = nd.reshape(feat, shape=(feat.shape[0], -1, feat.shape[3]))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return mxnet.symbol.flip(x, axis=3)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return nd.array(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return nd.array(tmp.reshape(shape)).to(x.device)


def symbolic_gather_feat(F, feat, ind, K, attri=1, mask=None):
    batch = 1
    #print("In symbolic_gather_feat, feat.shape = ", feat.shape)
    #print("In symbolic_gather_feat, ind.shape = ", ind.shape)

    if attri == 1:
        data = feat.reshape((batch, -1))
        index = ind.reshape((batch, -1))
    else:
        data = feat.reshape((batch, attri, -1))
        data = F.swapaxes(data, 1, 2)
        index = ind.reshape((batch, -1))

    #print("In symbolic_gather_feat, data.shape = ", data.shape)
    #print("In symbolic_gather_feat, index.shape = ", index.shape)
    temp = F.take(data, index, axis = 1, mode='wrap')
    #print("In symbolic_gather_feat, temp.shape = ", temp.shape)

    '''
    aa = F.split(temp, axis=2, num_outputs= K)
    print(aa[0].shape)
    bb = [F.linalg.extractdiag(F.squeeze(item)) for item in aa]
    output = F.stack(*bb).expand_dims(2)
    print(output.shape)
    '''

    # or simply:
    if attri == 1:
        output= temp.reshape((batch, K))
    else:
        output= temp.reshape((batch, K, attri))
    return output

def symbolic_transpose_and_gather_feat(F, feat, ind, K, batch, cat, attri):
    #print("In symbolic_transpose_and_gather_feat, feat.shape = ", feat.shape)
    feat = F.transpose(feat, axes=(0, 2, 3, 1))
    feat = F.reshape(feat, shape=(batch, -1, cat))
    #print("In symbolic_transpose_and_gather_feat, feat.shape = ", feat.shape)

    feat = symbolic_gather_feat(F, feat, ind, K, attri)
    return feat
