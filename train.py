"""
Author: Guanghan Ning
Date:   August, 2019
"""
import sys, os, time
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/dataset")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/dataset")

import mxnet as mx
from mxnet import nd, gluon, init, autograd
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad

from opts import opts

from models.model import create_model, load_model, save_model
from models.hourglass import stacked_hourglass
from models.decoder import decode_centernet
from models.losses import CtdetLoss

from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric  # https://gluon-cv.mxnet.io/_modules/gluoncv/utils/metrics/coco_detection.html

from coco_centernet import CenterCOCODataset

def get_coco(opt, coco_path="/export/guanghan/coco"):
    """Get coco dataset."""
    train_dataset = CenterCOCODataset(opt, split = 'train')   # custom dataset
    #train_dataset = CenterCOCODataset(opt, split = 'val')   # custom dataset
    val_dataset = gdata.COCODetection(root= coco_path, splits='instances_val2017', skip_empty=False)  # gluon official
    eval_metric = COCODetectionMetric(val_dataset,
                                     save_prefix = '_eval',
                                     #use_time = False,
                                     #cleanup= True,
                                     #score_thresh = 0,
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

    for epoch in range(opt.cur_epoch, opt.num_epochs):
        # training loop
        print("Training Epoch: {}".format(epoch))
        cumulative_train_loss = nd.zeros(1, ctx=ctx[0])
        training_samples = 0

        start = time.time()
        for i, batch in enumerate(train_loader):
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
                print("\t Iter: {}, loss: {}".format(i, losses[0].as_in_context(ctx[0]).asscalar()))

        train_hours = (time.time() - start) / 3600.0 # 1 epoch training time in hours
        train_loss_per_epoch = cumulative_train_loss.asscalar() / training_samples
        print("Epoch {}, time: {:.1f}, training loss: {:.2f}".format(epoch, train_hours, train_loss_per_epoch))

        # Save parameters
        prefix = "CenterNet_" + opt.arch
        save_model(model, '{:s}_{:04d}.params'.format(prefix, epoch))

        # validation loop
        if epoch % opt.val_interval != 0: continue
        map_name, mean_ap = validate(model, val_loader, ctx, eval_metric)
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))


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
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            pred = model(x)
            heatmaps, scale, offset = pred[-1]["hm"], pred[-1]["wh"], pred[-1]["reg"]  # last stack of hourglass result

            bboxes, scores, ids = decode_centernet(heat=heatmaps, wh=scale, reg=offset, flag_split=True)

            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


if __name__ == "__main__":
    opt = opts().init()
    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    print("Using Devices: ", ctx)

    """ 1. network """
    print('Creating model...')
    print("Using network architecture: ", opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    if opt.flag_finetune:
        model = load_model(model, opt.pretrained_path, ctx = ctx)
        opt.cur_epoch = int(opt.pretrained_path.split('.')[0][-4:])
    else:
        opt.cur_epoch = 0
        model.collect_params().initialize(init=init.Xavier(), ctx = ctx)

    """ 2. Dataset """
    train_dataset, val_dataset, eval_metric = get_coco(opt, "./data/coco")
    data_shape = opt.input_res
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    train_loader, val_loader = get_dataloader(train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx)

    """ 3. Training """
    train(model, train_loader, val_loader, eval_metric, ctx, opt)
