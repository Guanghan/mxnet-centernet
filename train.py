"""
Author: Guanghan Ning
Date:   August, 2019
"""
from models.model import create_model, load_model, save_model
from opts import opts

from models.large_hourglass import stacked_hourglass
from mxnet import nd, gluon, init
import mxnet as mx

from gluoncv import data as gdata

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

    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    # image transformer
    img_transform = transforms.Compose([transforms.Resize(300),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomBrightness(0.1),
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
        cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
        training_samples = 0

        for i, batch in enumerate(train_loader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = model(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)

                sum_loss, cls_loss, box_loss = criterion(cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)

            cumulative_train_loss += sum_loss.sum()
            training_samples += data.shape[0]
        train_loss_per_epoch = cumulative_train_loss.asscalar() / training_samples
        print("Epoch {}, training loss: {:.2f}".format(epoch, train_loss_per_epoch))

        # validation loop
        if epoch % self.val_interval != 0: continue
        cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
        valid_samples = 0

        map_name, mean_ap = validate(model, val_loader, ctx, eval_metric)
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))

        # Save parameters
        prefix = "CenterNet_" + opt.arch
        save_model(model, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch))


def validate(model, val_loader, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    model.set_nms(nms_thresh=0.45, nms_topk=400)
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
            ids, scores, bboxes = model(x)
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
