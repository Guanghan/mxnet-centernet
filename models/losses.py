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
from models.tensor_utils import _tranpose_and_gather_feat

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).astype('float32')
  neg_inds = gt.lt(1).astype('float32')

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
    pos_inds = gt.eq(1).astype('float32')
    neg_inds = gt.lt(1).astype('float32')
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
    mask = mask.unsqueeze(2).expand_as(pred).astype('float32')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Block):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).astype('float32')
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
    mask = mask.astype('float32')
    loss = nd.abs(pred*mask - target*mask).sum()
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Block):
  def __init__(self):
    super(L1Loss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).astype('float32')
    loss = nd.abs(pred*mask - target*mask).mean()
    return loss


class CtdetLoss_with_dict(nn.Block):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = gluon.loss.L2Loss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = gluon.loss.L1Loss(reduction='sum') if opt.dense_wh else \   # why sum? should be mean right?
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      # 1. heatmap loss
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

      # 2. scale loss
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) /
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks

      # 3. offset loss
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks

    # total loss
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats


class CtdetLoss(nn.Block):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = gluon.loss.L2Loss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  #def forward(self, outputs, batch):
  def criterion(outputs, targets_heatmaps, targets_scale, targets_offset, targets_inds, targets_reg_mask):
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
          targets_scale.detach().cpu().numpy(),
          targets_inds.detach().cpu().numpy(),
          output['wh'].shape[3], output['wh'].shape[2])).as_in_context(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = nd.from_numpy(gen_oracle_map(
          targets_offset.detach().cpu().numpy(),
          targets_inds.detach().cpu().numpy(),
          output['reg'].shape[3], output['reg'].shape[2])).as_in_context(opt.device)

      # 1. heatmap loss
      hm_loss += self.crit(output['hm'], targets_heatmaps) / opt.num_stacks

      # 2. scale loss
      if opt.wh_weight > 0:
        wh_loss += self.crit_reg(
            output['wh'], targets_reg_mask,
            targets_inds, targets_scale) / opt.num_stacks

      # 3. offset loss
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], targets_reg_mask,
                             targets_inds, targets_offset) / opt.num_stacks

    # total loss
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    #return loss, loss_stats
    return hm_loss, wh_loss, off_loss
