"""
Author: Guanghan Ning
Date:   August, 2019
"""
from mxnet import nd, gluon, init
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad

from opts import opts

from models.model import create_model, load_model, save_model
from models.large_hourglass import stacked_hourglass
from model.decoder import decode_centernet


def get_coco(coco_path="/export/guanghan/coco"):
    """Get coco dataset."""
    train_dataset = gdata.COCODetection(root= coco_path, splits='instances_train2017')
    val_dataset = gdata.COCODetection(root= coco_path, splits='instances_val2017', skip_empty=False)
    eval_metric = COCODetectionMetric(val_dataset,
                                     '_eval',
                                     cleanup=True,
                                     data_shape=(args.data_shape, args.data_shape))
    # coco validation is slow, consider increase the validation interval
    args.val_interval = 10
    return train_dataset, val_dataset, eval_metric


def get_dataloader(train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape

    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack())  # stack image, heatmaps, scale, offset
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    # image transformer
    img_transform = transforms.Compose([transforms.Resize(640),
                                        transforms.RandomResizedCrop(512, scale=(0.6, 1.3), ratio=(0.75, 1.33)),
                                        transforms.RandomFlipLeftRight(),
                                        transfroms.RandomColorJitter(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0, 1)])

    train_data = train_dataset.transform_first(img_transform)
    train_loader = gluon.data.DataLoader( train_data,
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    val_data = val_dataset.transform_first(img_transform)
    val_loader = gluon.data.DataLoader( val_data,
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def train(model, train_loader, val_loader, eval_metric, ctx, args):
    """Training pipeline"""
    model.collect_params().reset_ctx(ctx)

    trainer = gluon.Trainer(model.collect_params(),
                           'sgd',
                           {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    for epoch in range(args.start_epoch, args.epochs):
        # training loop
        cumulative_train_loss = nd.zeros(1, ctx=ctx)
        training_samples = 0

        for i, batch in enumerate(train_loader):
            X = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            targets_heatmaps = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)  # heatmaps: (batch, num_classes, H/S, W/S)
            targets_scale = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)  # scale: wh (batch, 2, H/S, W/S)
            targets_offset = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0) # offset: xy (batch, 2, H/s, W/S)

            with autograd.record():
                Y = model(X)
                preds_heatmaps, preds_scale, preds_offset = Y[0]["hm"], Y[0]["wh"], Y[0]["reg"]

                heatmap_crossentropy_focal_loss, scale_L1_loss, offset_L1_loss = criterion(targets_heatmaps, targets_scale, targets_offset,
                                                                                           preds_heatmaps, preds_scale, preds_offset)
                sum_loss = heatmap_crossentropy_focal_loss + 0.1 * scale_L1_loss + 1.0 * offset_L1_loss
                autograd.backward(sum_loss)
            # normalize loss by batch-size
            trainer.step(X.shape[0])

            cumulative_train_loss += sum_loss.sum()
            training_samples += data.shape[0]
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
            heatmaps, scale, offset = pred[0]["hm"], pred[0]["wh"], pred[0]["reg"]
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
    """ 1. network """
    print('Creating model...')
    opt = opts().init()
    print("Using network architecture: ", opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.collect_params().initialize(init=init.Xavier())

    """ 2. Dataset """
    train_dataset, val_dataset, eval_metric = get_coco("./data/coco")
    train_loader, val_loader = get_dataloader(train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx)

    """ 3. Training """
    train(model, train_loader, val_loader, eval_metric, ctx, args)
