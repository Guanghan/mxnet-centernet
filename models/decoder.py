"""
Author: Guanghan Ning
Date:   August, 2019

Adapted from the official Pytorch implementation:
https://github.com/xingyizhou/CenterNet

Homage to [PyTorch to Mxnet cheetsheet]:
https://gist.github.com/zhanghang1989/3d646f71d60c17048cf8ad582393ac6c
"""
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

from models.tensor_utils import _gather_feat, _tranpose_and_gather_feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nd.Pooling(data=heat, kernel= (kernel, kernel), stride=(1,1), pad=(pad,pad)) # default is max pooling
    keep = (hmax == heat).astype('float32')
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.shape

    [topk_scores, topk_inds] = nd.topk(nd.reshape(scores, (batch, cat, -1)), ret_typ='both', k=K)  # return both value and indices

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).astype('int').astype('float32')
    topk_xs   = (topk_inds % width).astype('int').astype('float32')

    [topk_score, topk_ind] = nd.topk(nd.reshape(topk_scores, (batch, -1)), ret_typ='both', k=K)
    topk_clses = (topk_ind / K).astype('int')

    topk_inds = _gather_feat(nd.reshape(topk_inds, (batch, -1, 1)), topk_ind)
    topk_inds = nd.reshape(topk_inds, (batch, K))

    topk_ys = _gather_feat(nd.reshape(topk_ys, (batch, -1, 1)), topk_ind)
    topk_ys = nd.reshape(topk_ys, (batch, K))

    topk_xs = _gather_feat(nd.reshape(topk_xs, (batch, -1, 1)), topk_ind)
    topk_xs = nd.reshape(topk_xs, (batch, K))

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_centernet(heat, wh, reg=None, cat_spec_wh=False, K=100, flag_split=False):
    batch, cat, height, width = heat.shape

    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = nd.reshape(reg, (batch, K, 2))
        xs = nd.reshape(xs, (batch, K, 1)) + reg[:, :, 0:1]
        ys = nd.reshape(ys, (batch, K, 1)) + reg[:, :, 1:2]
    else:
        xs = nd.reshape(xs, (batch, K, 1)) + 0.5
        ys = nd.reshape(ys, (batch, K, 1)) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = nd.reshape(wh, (batch, K, cat, 2))
        clses_ind = nd.reshape(clses, (batch, K, 1, 1))

        clses_ind = nd.stack(clses_ind, clses_ind, axis=3)   #becomes (batch, K, 1, 2)
        clses_ind = clses_ind.astype('int64')

        wh = wh.gather_nd(2, clses_ind)
        wh = nd.reshape(wh, (batch, K, 2))
    else:
        wh = nd.reshape(wh, (batch, K, 2))

    clses  = nd.reshape(clses, (batch, K, 1)).astype('float32')
    scores = nd.reshape(scores, (batch, K, 1))

    bboxes =  nd.concat(xs - wh[:, :, 0:1] / 2,
                        ys - wh[:, :, 1:2] / 2,
                        xs + wh[:, :, 0:1] / 2,
                        ys + wh[:, :, 1:2] / 2,
                        dim=2)

    if flag_split is True:
        return bboxes, scores, clses
    else:
        detections = nd.concat(bboxes, scores, clses, dim=2)
        return detections


if __name__ == "__main__":
    scores = nd.random.uniform(shape=(2,3,128,128))
    batch, cat, height, width = scores.shape
    [topk_scores, topk_inds] = nd.topk(nd.reshape(scores, (batch, cat, -1)), ret_typ='both', k=2)  # return both value and indices
    print(topk_scores)
    print(topk_inds)

    topk_score, topk_inds, topk_clses, topk_ys, topk_xs = _topk(scores, K=4)
    print(topk_score)
