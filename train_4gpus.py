"""
Author: Guanghan Ning
Date:   August, 2019
"""
import sys, os
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/dataset")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/dataset")

import mxnet as mx
from mxnet import nd, gluon, init, autograd
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad

from opts import opts

from models.model import create_model, load_model, save_model
from models.large_hourglass import stacked_hourglass
from models.decoder import decode_centernet
from models.losses import CtdetLoss

from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

from coco_centernet import CenterCOCODataset

def get_coco(opt, coco_path="/export/guanghan/coco"):
    """Get coco dataset."""
    train_dataset = CenterCOCODataset(opt, split = 'train')   # custom dataset
    val_dataset = gdata.COCODetection(root= coco_path, splits='instances_val2017', skip_empty=False)  # gluon official
    eval_metric = COCODetectionMetric(val_dataset,
                                     '_eval',
                                     cleanup=True,
                                     data_shape=(opt.input_res, opt.input_res))
    # coco validation is slow, consider increase the validation interval
    opt.val_interval = 10
    return train_dataset, val_dataset, eval_metric


def get_dataloader(train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape

    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack(), Stack(), Stack())  # stack image, heatmaps, scale, offset, inds, masks
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    train_loader = gluon.data.DataLoader( train_dataset,
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def train(model, train_loader, val_loader, eval_metric, ctx, opt):
    """Training pipeline"""
    model.collect_params().reset_ctx(ctx)

    trainer = gluon.Trainer(model.collect_params(),
                           'adam',
                           {'learning_rate': opt.lr})
    criterion = CtdetLoss(opt)

    for epoch in range(0, opt.num_epochs):
        # training loop
        cumulative_train_loss = nd.zeros(1, ctx=ctx[0])
        training_samples = 0

        for i, batch in enumerate(train_loader):
            #print("Iter: {}".format(i))

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            targets_heatmaps = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)  # heatmaps: (batch, num_classes, H/S, W/S)
            targets_scale = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)  # scale: wh (batch, 2, H/S, W/S)
            targets_offset = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0) # offset: xy (batch, 2, H/s, W/S)
            targets_inds = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)
            targets_mask = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                losses = [ criterion(model(X), hm, scale, offset, inds, mask) \
                for X, hm, scale, offset, inds, mask in zip(data, targets_heatmaps, targets_scale, targets_offset, targets_inds, targets_mask) ]

            for loss in losses:
                loss.backward()

            # normalize loss by batch-size
            num_gpus = len(opt.gpus)
            trainer.step(opt.batch_size // num_gpus, ignore_stale_grad=True)

            for loss in losses:
                cumulative_train_loss += loss.sum().as_in_context(ctx[0])
                training_samples += opt.batch_size // num_gpus

            if i % 200 == 1:
                print("Iter: {}, loss: {}".format(i, losses[0].as_in_context(ctx[0])))

        train_loss_per_epoch = cumulative_train_loss.asscalar() / training_samples
        print("Epoch {}, training loss: {:.2f}".format(epoch, train_loss_per_epoch))

        # validation loop
        if epoch % self.val_interval != 0: continue
        map_name, mean_ap = validate(model, val_loader, ctx, eval_metric)
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))

        # Save parameters
        prefix = "CenterNet_" + opt.arch
        save_model(model, '{:s}_{:04d}.params'.format(prefix, epoch))


def validate(model, val_loader, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    for batch in val_loader:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        for x, y in zip(data, label):
            # get prediction results
            pred = model(x)
            #heatmaps, scale, offset = pred[0]["hm"], pred[0]["wh"], pred[0]["reg"]  # 1st stack hourglass result
            heatmaps, scale, offset = pred[1]["hm"], pred[1]["wh"], pred[1]["reg"]  # 2nd stack hourglass result
            bboxes, scores, ids = decode_centernet(heat=heatmaps, wh=scale, reg=offset, flag_split=True)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


# call:
# python train.py ctdet --arch hourglass
if __name__ == "__main__":
    opt = opts().init()
    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    print("Using Devices: ", ctx)

    """ 1. network """
    print('Creating model...')
    print("Using network architecture: ", opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.collect_params().initialize(init=init.Xavier(), ctx = ctx)

    """ 2. Dataset """
    train_dataset, val_dataset, eval_metric = get_coco(opt, "./data/coco")
    data_shape = opt.input_res
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    train_loader, val_loader = get_dataloader(train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx)

    """ 3. Training """
    train(model, train_loader, val_loader, eval_metric, ctx, opt)
