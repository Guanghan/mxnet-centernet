"""
Author: Guanghan Ning
Date:   August, 2019
"""
import sys, os
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/")
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
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric  # https://gluon-cv.mxnet.io/_modules/gluoncv/utils/metrics/coco_detection.html

from coco_centernet import CenterCOCODataset

def get_coco(opt, coco_path="/export/guanghan/coco"):
    """Get coco dataset."""
    val_dataset = gdata.COCODetection(root= coco_path, splits='instances_val2017', skip_empty=False)  # gluon official
    eval_metric = COCODetectionMetric(val_dataset,
                                     save_prefix = '_eval',
                                     #use_time = False,
                                     #cleanup= True,
                                     #score_thresh = 0,
                                     data_shape=(opt.input_res, opt.input_res))
    return val_dataset, eval_metric


def get_dataloader(val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return val_loader


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
            #heatmaps, scale, offset = pred[0]["hm"], pred[0]["wh"], pred[0]["reg"]  # 1st stack hourglass result
            heatmaps, scale, offset = pred[1]["hm"], pred[1]["wh"], pred[1]["reg"]  # 2nd stack hourglass result

            #DEBUGING decode_centernet:
            bboxes, scores, ids = decode_centernet(heat=heatmaps, wh=scale, reg=offset, flag_split=True)

            #print("bboxes: {}, scores {}, ids {} \n".format(bboxes.shape, scores.shape, ids.shape))
            # shape: (6, 100, 4), (6, 100, 1), (6, 100, 1)
            # shape: (batch, K, attributes)
            print("Top-5, bbox: {}, scores: {}, id {} \n".format(bboxes[0, 0:5, :], scores[0, 0:5, :], ids[0, 0:5, :]))

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


def test_validation(model, val_loader, eval_metric, ctx, opt):
    # validation loop
    map_name, mean_ap = validate(model, val_loader, ctx, eval_metric)
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))


if __name__ == "__main__":
    opt = opts().init()
    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    print("Using Devices: ", ctx)

    """ 1. network """
    print('Creating model...')
    print("Using network architecture: ", opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    pretrained_path = "/root/CenterNet-Gluon/CenterNet_hourglass_0030.params"
    model = load_model(model, pretrained_path, ctx = ctx)

    """ 2. Dataset """
    val_dataset, eval_metric = get_coco(opt, "./data/coco")
    data_shape = opt.input_res
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    val_loader = get_dataloader(val_dataset, data_shape, batch_size, num_workers, ctx)

    """ 3. Testing Validation """
    test_validation(model, val_loader, eval_metric, ctx, opt)
