"""
Author: Guanghan Ning
Date:   August, 2019

Adapted from the official Pytorch implementation:
https://github.com/xingyizhou/CenterNet
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

    # perform nms on heatmaps, find the peaks
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


def decode_centernet_3dod(heat, rot, depth, dim, wh=None, reg=None, K=40):
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

    rot = _tranpose_and_gather_feat(rot, inds)
    rot = nd.reshape(rot, (batch, K, 8))
    depth = _tranpose_and_gather_feat(depth, inds)
    depth = nd.reshape(depth, (batch, K, 1))
    dim = _tranpose_and_gather_feat(dim, inds)
    dim = nd.reshape(dim, (batch, K, 3))

    clses  = nd.reshape(clses, (batch, K, 1)).astype('float32')
    scores = nd.reshape(scores, (batch, K, 1))
    xs = nd.reshape(xs, (batch, K, 1))
    ys = nd.reshape(ys, (batch, K, 1))

    if wh is not None:
        wh = _tranpose_and_gather_feat(wh, inds)
        wh = nd.reshape(wh, (batch, K, 2))
        detections = nd.concat(xs, ys, scores, rot, depth, dim, wh, clses, dim=2)
    else:
        detections = nd.concat(xs, ys, scores, rot, depth, dim, clses, dim=2)

    return detections


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.shape

    [topk_scores, topk_inds] = nd.topk(scores.reshape((batch, cat, -1)), ret_typ = "both", k= K)

    #[topk_score, topk_ind] = nd.topk(nd.reshape(topk_scores, (batch, -1)), ret_typ='both', k=K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).astype('int32').astype('float32')
    topk_xs   = (topk_inds % width).astype('int32').astype('float32')

    return topk_scores, topk_inds, topk_ys, topk_xs


def decode_centernet_pose(heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.shape
    num_joints = kps.shape[1] // 2
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _tranpose_and_gather_feat(kps, inds)
    kps = nd.reshape(kps, (batch, K, num_joints * 2))

    kps[:, :, ::2] += nd.reshape(xs, (batch, K, 1)).broadcast_to((batch, K, num_joints))
    kps[:, :, 1::2] += nd.reshape(ys, (batch, K, 1)).broadcast_to((batch, K, num_joints))

    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = nd.reshape(reg,(batch, K, 2))
        xs = xs.reshape((batch, K, 1)) + reg[:, :, 0:1]
        ys = ys.reshape((batch, K, 1)) + reg[:, :, 1:2]
    else:
        xs = xs.reshape((batch, K, 1)) + 0.5
        ys = ys.reshape((batch, K, 1)) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.reshape((batch, K, 2))
    clses  = clses.reshape((batch, K, 1)).astype('float32')
    scores = scores.reshape((batch, K, 1))

    bboxes =  nd.concat(xs - wh[:, :, 0:1] / 2,
                        ys - wh[:, :, 1:2] / 2,
                        xs + wh[:, :, 0:1] / 2,
                        ys + wh[:, :, 1:2] / 2,
                        dim=2)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.reshape((batch, K, num_joints, 2))
        kps = nd.swapaxes(kps, 1, 2) # b x J x K x 2

        reg_kps = nd.expand_dims(kps, axis=3).broadcast_to((batch, num_joints, K, K, 2))
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K

        if hp_offset is not None:
            hp_offset = _tranpose_and_gather_feat(hp_offset, hm_inds.reshape((batch, -1)))
            hp_offset = hp_offset.reshape((batch, num_joints, K, 2))
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).astype('float32')
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs

        hm_kps = nd.stack(hm_xs, hm_ys, axis=-1).expand_dims(axis=2).broadcast_to((batch, num_joints, K, K, 2))
        dist = (((reg_kps - hm_kps) ** 2).sum(axis=4) ** 0.5)
        min_dist = dist.min(axis=3) # b x J x K
        min_ind = nd.argmin(dist, axis=3) # b x J x K

        M, N, K = hm_score.shape[0:3]
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    hm_score[i, j, k] = hm_score[i, j, min_ind[i, j, k]]
        hm_score = hm_score.expand_dims(axis=-1)
        min_dist = min_dist.expand_dims(-1)

        hm_kps = hm_kps.reshape((batch, num_joints, K, 2))
        for i in range(M):
            for j in range(N):
                for k in range(K):
                        hm_kps[i, j, k, 0] = hm_kps[i, j, min_ind[i, j, k], 0]
                        hm_kps[i, j, k, 1] = hm_kps[i, j, min_ind[i, j, k], 1]

        l = bboxes[:, :, 0].reshape((batch, 1, K, 1)).broadcast_to((batch, num_joints, K, 1))
        t = bboxes[:, :, 1].reshape((batch, 1, K, 1)).broadcast_to((batch, num_joints, K, 1))
        r = bboxes[:, :, 2].reshape((batch, 1, K, 1)).broadcast_to((batch, num_joints, K, 1))
        b = bboxes[:, :, 3].reshape((batch, 1, K, 1)).broadcast_to((batch, num_joints, K, 1))

        mask = (hm_kps[:, :, 0:1] < l) + (hm_kps[:, :, 0:1] > r)
        mask += (hm_kps[:, :, 1:2] < t) + (hm_kps[:, :, 1:2] > b)
        mask += (hm_score < thresh)
        mask += (min_dist > (nd.maximum(b - t, r - l) * 0.3))
        mask = (mask > 0).astype('float32').broadcast_to((batch, num_joints, K, 2))

        kps = (1 - mask) * hm_kps + mask * kps
        kps = nd.swapaxes(kps, 1, 2).reshape((batch, K, num_joints * 2))

    detections = nd.concat(bboxes, scores, kps, clses, dim=2)
    return detections


if __name__ == "__main__":
    scores = nd.random.uniform(shape=(2,3,128,128))
    batch, cat, height, width = scores.shape
    [topk_scores, topk_inds] = nd.topk(nd.reshape(scores, (batch, cat, -1)), ret_typ='both', k=2)  # return both value and indices
    print(topk_scores)
    print(topk_inds)

    topk_score, topk_inds, topk_clses, topk_ys, topk_xs = _topk(scores, K=4)
    print(topk_score)
