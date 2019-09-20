# ------------------------------------------------------------------------------
# Guanghan Ning
# Sep, 2019
# Portions of this code are adpated from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon, init, nd

from utils.oracle_utils import gen_oracle_map
from models.tensor_utils import _tranpose_and_gather_feat, _sigmoid

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.__eq__(1).astype('float32')
  neg_inds = gt.__lt__(1).astype('float32')

  neg_weights = nd.power(1 - gt, 4)

  loss = 0

  pos_loss = nd.log(pred) * nd.power(1 - pred, 2) * pos_inds
  neg_loss = nd.log(1 - pred) * nd.power(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.astype('float32').sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.__eq__(1).astype('float32')
    neg_inds = gt.__lt__(1).astype('float32')
    num_pos  = pos_inds.astype('float32').sum()
    neg_weights = nd.power(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = nd.log(1 - trans_pred) * nd.power(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]

    t = nd.abs(regr - gt_regr)
    regt_loss = smooth_l1(t).sum()
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.astype('float32').sum()
  mask = mask.expand_dims(2).broadcast_like(gt_regr).astype('float32')

  regr = regr * mask
  gt_regr = gt_regr * mask

  t = nd.abs(regr - gt_regr)
  regt_loss = smooth_l1(t).sum()
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Block):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Block):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Block):
  def __init__(self):
    super(RegL1Loss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    pred = pred.swapaxes(dim1 = 0, dim2 = 1)
    mask = mask.expand_dims(axis = 2).broadcast_like(pred).astype('float32')
    loss = nd.abs(pred*mask - target*mask).sum()
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Block):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    pred = pred.swapaxes(dim1 = 0, dim2 = 1)
    mask = mask.expand_dims(axis = 2).broadcast_like(pred).astype('float32')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = nd.abs(pred*mask - target*mask).sum()
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Block):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    pred = pred.swapaxes(dim1 = 0, dim2 = 1)
    mask = mask.astype('float32')
    loss = nd.abs(pred*mask - target*mask).sum()
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Block):
  def __init__(self):
    super(L1Loss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    pred = pred.swapaxes(dim1 = 0, dim2 = 1)
    mask = mask.expand_dims(axis = 2).broadcast_like(pred).astype('float32')
    loss = nd.abs(pred*mask - target*mask).mean()
    return loss

'''
Loss for 2DOD
'''
class CtdetLoss(nn.Block):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = gluon.loss.L2Loss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = gluon.loss.L1Loss() if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, targets_heatmaps, targets_scale, targets_offset, targets_inds, targets_reg_mask):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      # Optional: Use ground truth for validation
      if opt.eval_oracle_hm:
        output['hm'] = targets_heatmaps
      if opt.eval_oracle_wh:
        output['wh'] = nd.from_numpy(gen_oracle_map(
          targets_scale.asnumpy(),
          targets_inds.asnumpy(),
          output['wh'].shape[3], output['wh'].shape[2])).as_in_context(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = nd.from_numpy(gen_oracle_map(
          targets_offset.asnumpy(),
          targets_inds.asnumpy(),
          output['reg'].shape[3], output['reg'].shape[2])).as_in_context(opt.device)

      # 1. heatmap loss
      hm_loss = hm_loss + self.crit(output['hm'], targets_heatmaps) / opt.num_stacks

      # 2. scale loss
      if opt.wh_weight > 0:
        wh_loss = wh_loss + self.crit_reg(
            output['wh'], targets_reg_mask,
            targets_inds, targets_scale) / opt.num_stacks

      # 3. offset loss
      if opt.reg_offset and opt.off_weight > 0:
        off_loss = off_loss + self.crit_reg(output['reg'], targets_reg_mask,
                             targets_inds, targets_offset) / opt.num_stacks

    # total loss
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss


'''
Loss for 3DOD
'''
class BinRotLoss(nn.Block):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

def compute_res_loss(output, target, maski):
    diff = nd.abs(output * maski - target * maski)
    return nd.smooth_l1(diff).mean()

def compute_bin_loss(output, target, mask):
    mask = mask.broadcast_like(output)
    output = output * mask.astype('float32')
    return nd.softmax_cross_entropy(output, target).mean() / (1.0*output.size)

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = nd.reshape(output, (-1, 8))
    target_bin = nd.reshape(target_bin, (-1, 2))
    target_res = nd.reshape(target_res, (-1, 2))
    mask = nd.reshape(mask, (-1, 1))
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = nd.zeros_like(loss_bin1)

    mask1 = (target_bin[:, 0] > 0).astype('float32')
    if mask1.sum() > 0:
        valid_output1 = output
        valid_target_res1 = target_res

        loss_sin1 = compute_res_loss(valid_output1[:, 2], nd.sin(valid_target_res1[:, 0]), mask1)
        loss_cos1 = compute_res_loss(valid_output1[:, 3], nd.cos(valid_target_res1[:, 0]), mask1)
        loss_res = loss_res + loss_sin1 + loss_cos1

    mask2 = (target_bin[:, 1] > 0).astype('float32')
    if mask2.sum() > 0:
        valid_output2 = output
        valid_target_res2 = target_res

        loss_sin2 = compute_res_loss(valid_output2[:, 6], nd.sin(valid_target_res2[:, 1]), mask2)
        loss_cos2 = compute_res_loss(valid_output2[:, 7], nd.cos(valid_target_res2[:, 1]), mask2)
        loss_res = loss_res + loss_sin2 + loss_cos2
    #print("loss_bin1: {}, loss_bin2: {}, loss_sin1: {}, loss_sin2: {}, loss_cos1: {}, loss_cos2: {}".format(loss_bin1, loss_bin2, loss_sin1, loss_sin2, loss_cos1, loss_cos2))
    return loss_bin1 + loss_bin2 + loss_res


class DddLoss(nn.Block):
    def __init__(self, opt):
        super(DddLoss, self).__init__()
        # Criterion: MSE loss or focal loss
        self.crit = gluon.loss.L2Loss() if opt.mse_loss else FocalLoss()
        # Criterion for regression: L1 loss
        self.crit_reg = L1Loss()
        # Criterion for rotation: novel Bin Rotation Loss
        self.crit_rot = BinRotLoss()
        self.opt = opt

    '''
    dep: depth
    hm: orientation 1st part, classification to decide which bin, 2x2 values (c1, c2, 2bins)
    ind: orientation 2nd part, L1 loss for the angular values, 2x2 values (sin, cos, 2bins)
    '''
    def forward(self, outputs, \
                targets_center, targets_2d_wh, targets_2d_offset, \
                targets_3d_depth, targets_3d_dim, targets_3d_rotbin, targets_3d_rotres, \
                targets_inds, targets_2d_wh_mask, targets_3d_rot_mask):
        opt = self.opt

        hm_loss, dep_loss, rot_loss, dim_loss = 0, 0, 0, 0
        wh_loss, off_loss = 0, 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.

            if opt.eval_oracle_dep:
                output['dep'] = nd.array(gen_oracle_map(
                                                 targets_3d_depth.asnumpy(),
                                                 targets_inds.asnumpy(),
                                                 opt.output_w, opt.output_h)).as_in_context(opt.device)

            hm_loss = hm_loss + self.crit(output['hm'], targets_center) / opt.num_stacks
            if opt.dep_weight > 0:
                dep_loss = dep_loss + self.crit_reg(output['dep'], targets_2d_wh_mask,
                                          targets_inds, targets_3d_depth) / opt.num_stacks
            if opt.dim_weight > 0:
                dim_loss = dim_loss + self.crit_reg(output['dim'], targets_2d_wh_mask,
                                          targets_inds, targets_3d_dim) / opt.num_stacks
            if opt.rot_weight > 0:
                rot_loss = rot_loss + self.crit_rot(output['rot'], targets_3d_rot_mask,
                                          targets_inds, targets_3d_rotbin,
                                          targets_3d_rotres) / opt.num_stacks
            if opt.reg_bbox and opt.wh_weight > 0:
                wh_loss = wh_loss + self.crit_reg(output['wh'], targets_3d_rot_mask,
                                         targets_inds, targets_2d_wh) / opt.num_stacks
            if opt.reg_offset and opt.off_weight > 0:
                off_loss = off_loss + self.crit_reg(output['reg'], targets_3d_rot_mask,
                                          targets_inds, targets_2d_offset) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.dep_weight * dep_loss + \
               opt.dim_weight * dim_loss + opt.rot_weight * rot_loss + \
               opt.wh_weight * wh_loss + opt.off_weight * off_loss

        #print("hm_loss: {}, dep_loss: {}, dim_loss: {}, rot_loss: {}, wh_loss: {}, off_loss: {}".format(hm_loss, dep_loss, dim_loss, rot_loss, wh_loss, off_loss))

        #loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'dep_loss': dep_loss,
        #              'dim_loss': dim_loss, 'rot_loss': rot_loss,
        #              'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss
