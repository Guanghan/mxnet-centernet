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
  y = mxnet.symbol.clip(data=x.sigmoid_(), a_min=1e-4, a_max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.shape[2]
    ind = ind.expand_dims(2).broadcast_to((ind.shape[0], ind.shape[1], dim))
    print("ind_new.shape = ", ind.shape)

    ind_identity = nd.zeros(shape= ind.shape)
    for i in range(ind_identity.shape[0]):
        ind_identity[i, :, :] = i
    ind_stack = nd.stack(ind_identity, ind, axis=0)
    print("ind_stack.shape = ", ind_stack.shape)

    feat = nd.gather_nd(data=feat, indices=ind_stack)  # something might be wrong here, probably should not use ind_stack
    # shape should be (2, 4)

    if mask is not None:
        mask = nd.expand_dims(mask, 2).broadcast_to(feat.shape)
        feat = feat[mask]
        feat = feat.reshape((-1, dim))
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = nd.transpose(feat, axes=(0, 2, 3, 1))
    feat = nd.reshape(feat, shape=(feat.shape[0], -1, feat.shape[3]))
    print("feat.shape = ", feat.shape)
    print("ind.shape = ", ind.shape)
    feat = _gather_feat(feat, ind)
    print("feat_after_gather.shape = ", feat.shape)
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
