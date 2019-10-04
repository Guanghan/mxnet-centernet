"""
Author: Guanghan Ning
Date:   August, 2019
"""
import sys, os, time
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/dataset")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/dataset")

import mxnet as mx
from mxnet import nd, gluon, init, autograd
from gluoncv.data.batchify import Tuple, Stack, Pad

from opts import opts
from models.model import create_model, load_model, save_model
from models.losses import DddLoss

from kitti_centernet import CenterKITTIDataset
from detectors.ddd_detector import DddDetector

from progress.bar import Bar
from utils.misc import AverageMeter

def get_kitti(opt, kitti_path):
    """Get coco dataset."""
    train_dataset = CenterKITTIDataset(opt, split = 'train')   # custom dataset
    val_dataset = CenterKITTIDataset(opt, split = 'val')   # custom dataset
    opt.val_interval = 10
    return train_dataset, val_dataset


def get_dataloader(train_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape

    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack())
    train_loader = gluon.data.DataLoader(train_dataset, batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader


def train(model, train_loader, val_dataset, ctx, opt):
    """Training pipeline"""
    model.collect_params().reset_ctx(ctx)

    trainer = gluon.Trainer(model.collect_params(),
                           'adam',
                           {'learning_rate': opt.lr})
    criterion = DddLoss(opt)

    for epoch in range(opt.cur_epoch, opt.num_epochs):
        # training loop
        print("Training Epoch: {}".format(epoch))
        cumulative_train_loss = nd.zeros(1, ctx=ctx[0])
        training_samples = 0

        start = time.time()
        for i, batch in enumerate(train_loader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            targets_center = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)  # heatmaps: (batch, num_classes, H/S, W/S)
            targets_2d_wh = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)  # scale: wh (batch, 2, H/S, W/S)
            targets_2d_offset = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0) # offset: xy (batch, 2, H/s, W/S)

            targets_3d_depth = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)
            targets_3d_dim = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)
            targets_3d_rotbin = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            targets_3d_rotres = gluon.utils.split_and_load(batch[7], ctx_list=ctx, batch_axis=0)

            targets_inds = gluon.utils.split_and_load(batch[8], ctx_list=ctx, batch_axis=0)
            targets_2d_wh_mask = gluon.utils.split_and_load(batch[9], ctx_list=ctx, batch_axis=0)
            targets_3d_rot_mask = gluon.utils.split_and_load(batch[10], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                losses = [criterion(model(X), hm, wh, offset, dep, dim, rotbin, rotres, inds, wh_mask, rot_mask) \
                for X, hm, wh, offset, dep, dim, rotbin, rotres, inds, wh_mask, rot_mask in \
                zip(data, targets_center, targets_2d_wh, targets_2d_offset, \
                    targets_3d_depth, targets_3d_dim, targets_3d_rotbin, targets_3d_rotres, \
                    targets_inds, targets_2d_wh_mask, targets_3d_rot_mask)]

            for loss in losses:
                loss.backward()

            # normalize loss by batch-size
            num_gpus = len(opt.gpus)
            trainer.step(opt.batch_size // num_gpus, ignore_stale_grad=True)

            for loss in losses:
                cumulative_train_loss += loss.sum().as_in_context(ctx[0])
                training_samples += opt.batch_size // num_gpus

            if i % 20 == 1:
                print("\t Iter: {}, loss: {}".format(i, losses[0].as_in_context(ctx[0]).asscalar()))

        train_hours = (time.time() - start) / 3600.0 # 1 epoch training time in hours
        train_loss_per_epoch = cumulative_train_loss.asscalar() / training_samples
        print("Epoch {}, time: {:.2f} hours, training loss: {:.3f}".format(epoch, train_hours, train_loss_per_epoch))

        # Save parameters
        prefix = "3DOD_" + opt.arch
        model_path = '{:s}_{:04d}.params'.format(prefix, epoch)
        if not os.path.exists(model_path):
            save_model(model, '{:s}_{:04d}.params'.format(prefix, epoch))

        # validation loop
        if epoch % opt.val_interval == 0:
            validate(model, val_dataset, opt, ctx[-1])


def validate(model, dataset, opt, ctx):
    """Test on validation dataset."""
    detector = DddDetector(opt)
    detector.model = model

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    print("Reporting every 1000 images...")
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]  # Kitti has been transformed to COCO format
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        ret = detector.run(img_path, img_info['calib'])

        results[img_id] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                       ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        if ind % 1000 == 0:
            bar.next()
    bar.finish()
    val_dataset.run_eval(results = results, save_dir = './output/')


if __name__ == "__main__":
    opt = opts()
    opt.task = "ddd"
    opt = opt.init()

    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    print("Using Devices: ", ctx)

    """ 1. network """
    print('Creating model...')
    print("Using network architecture: ", opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv, ctx = ctx)

    opt.cur_epoch = 0
    if opt.flag_finetune:
        model = load_model(model, opt.pretrained_path, ctx = ctx)
        #model = model.load_parameters(opt.pretrained_path, ctx=ctx, ignore_extra=True, allow_missing = True)
        opt.cur_epoch = int(opt.pretrained_path.split('.')[0][-4:])
    elif opt.arch != "res_18":
        model.collect_params().initialize(init=init.Xavier(), ctx = ctx)

    """ 2. Dataset """
    train_dataset, val_dataset = get_kitti(opt, "./data/kitti")
    data_shape = opt.input_res
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    train_loader = get_dataloader(train_dataset, data_shape, batch_size, num_workers, ctx)

    """ 3. Training """
    train(model, train_loader, val_dataset, ctx, opt)
