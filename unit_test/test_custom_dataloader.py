import sys, os
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/dataset")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/dataset")
from coco_centernet import CenterCOCODataset
import gluoncv
from gluoncv.data.batchify import Tuple, Stack, Pad

def test_load():
    from opts import opts
    opt = opts().init()

    batch_size = 16
    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack())  # stack image, heatmaps, scale, offset
    num_workers = 2

    train_dataset = CenterCOCODataset(opt, split = 'train')
    train_loader = gluon.data.DataLoader( train_dataset,
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    for i, batch in enumerate(train_loader):
        print(batch.shape)

        X = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        targets_heatmaps = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)  # heatmaps: (batch, num_classes, H/S, W/S)
        targets_scale = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)  # scale: wh (batch, 2, H/S, W/S)
        targets_offset = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0) # offset: xy (batch, 2, H/s, W/S)

        print(x.shape)
        print(targets_heatmaps.shape)
        print(targets_scale.shape)
        print(targets_offset.shape)
    return


if __name__ == "__main__":
    test_load()
