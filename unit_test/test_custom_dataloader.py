import sys, os
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/dataset")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/dataset")
from coco_centernet import CenterCOCODataset
import mxnet as mx
from mxnet import nd, gluon, init
import gluoncv
from gluoncv.data.batchify import Tuple, Stack, Pad

def test_load():
    from opts import opts
    opt = opts().init()

    batch_size = 16
    #batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack())  # stack image, heatmaps, scale, offset
    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack(), Stack(), Stack())  # stack image, heatmaps, scale, offset, ind, mask
    num_workers = 2

    train_dataset = CenterCOCODataset(opt, split = 'train')
    train_loader = gluon.data.DataLoader( train_dataset,
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    for i, batch in enumerate(train_loader):
        print("{} Batch".format(i))
        print("image batch shape: ", batch[0].shape)
        print("heatmap batch shape", batch[1].shape)
        print("scale batch shape", batch[2].shape)
        print("offset batch shape", batch[3].shape)
        print("indices batch shape", batch[4].shape)
        print("mask batch shape", batch[5].shape)

        X = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        targets_heatmaps = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)  # heatmaps: (batch, num_classes, H/S, W/S)
        targets_scale = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)  # scale: wh (batch, 2, H/S, W/S)
        targets_offset = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0) # offset: xy (batch, 2, H/s, W/S)
        targets_inds = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)
        targets_mask = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)

        print("len(targets_heatmaps): ", len(targets_heatmaps))
        print("First item: image shape: ", X[0].shape)
        print("First item: heatmaps shape: ", targets_heatmaps[0].shape)
        print("First item: scalemaps shape: ", targets_scale[0].shape)
        print("First item: offsetmaps shape: ", targets_offset[0].shape)
        print("First item: indices shape: ", targets_inds[0].shape)
        print("First item: mask shape: ", targets_mask[0].shape)
    return


if __name__ == "__main__":
    test_load()
