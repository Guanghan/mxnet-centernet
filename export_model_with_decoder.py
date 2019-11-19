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
from models.losses import MultiPoseLoss

from cocohp_centernet import CenterMultiPoseDataset
from detectors.pose_detector import PoseDetector

from progress.bar import Bar
from utils.misc import AverageMeter

import warnings


def get_coco(opt, coco_path):
    """Get coco dataset."""
    train_dataset = CenterMultiPoseDataset(opt, split = 'train')   # custom dataset
    val_dataset = CenterMultiPoseDataset(opt, split = 'val')   # custom dataset
    opt.val_interval = 10
    return train_dataset, val_dataset


def get_dataloader(train_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape

    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack())
    train_loader = gluon.data.DataLoader(train_dataset, batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

def validate(model, dataset, opt, ctx):
    """Test on validation dataset."""
    detector = PoseDetector(opt)
    detector.model = model

    results = {}
    num_iters = len(dataset)

    for ind in range(2):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        ret = detector.run(img_path)
        results[img_id] = ret['results']

    #print(model)
    print("Calling hybridize")
    detector.model.hybridize()
    #print(model)


    for ind in range(2):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        ret = detector.run(img_path)
        results[img_id] = ret['results']

    print("Save Mode: symbolic")
    prefix = "2DPose_{}_with_decode".format(opt.arch)
    epoch = 999

    #detector.save_symbols(img_path)
    model.export(prefix, epoch)

def test_hybridize(model, ctx):
    x = nd.random.normal(shape=(1, 3, 512, 512)).as_in_context(ctx)
    model(x)

    model.hybridize()
    model(x)
    return


if __name__ == "__main__":
    opt = opts()
    opt.task = "multi_pose"
    opt = opt.init()

    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    print("Using Devices: ", ctx)

    """ 1. network """
    print('Creating model...')
    print("Using network architecture: ", opt.arch)

    if opt.mode == "symbolic":
        print("Mode: symbolic")
        if opt.flag_finetune:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt.cur_epoch = int(opt.pretrained_path.split('.')[0][-4:])
                params_path = opt.pretrained_path
                json_path = opt.pretrained_path[:-11] + "symbol.json"
                model = gluon.nn.SymbolBlock.imports(json_path, ['data'], params_path, ctx=ctx)
        else:
            opt.cur_epoch = 0
            model = create_model(opt.arch, opt.heads, opt.head_conv, ctx = ctx)
            #model.hybridize()
            #print(model)
    else:
        print("Mode: imperative")
        opt.cur_epoch = 0
        model = create_model(opt.arch, opt.heads, opt.head_conv, ctx = ctx)
        if opt.flag_finetune:
            model = load_model(model, opt.pretrained_path, ctx = ctx)
            #model = model.load_parameters(opt.pretrained_path, ctx=ctx, ignore_extra=True, allow_missing = True)
            opt.cur_epoch = int(opt.pretrained_path.split('.')[0][-4:])
        elif opt.arch != "res_18":
            model.collect_params().initialize(init=init.Xavier(), ctx = ctx)


    """ 2. Dataset """
    train_dataset, val_dataset = get_coco(opt, "./data/coco")
    data_shape = opt.input_res
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    train_loader = get_dataloader(train_dataset, data_shape, batch_size, num_workers, ctx)

    """ 3. Training """
    #train(model, train_loader, val_dataset, ctx, opt)
    #validate(model, val_dataset, opt, ctx[-1])
    #test_hybridize(model, ctx[-1])

    x = nd.random.normal(shape=(1, 3, 512, 512)).as_in_context(ctx[-1])
    print(model(x))

    model.hybridize()
    print(model(x))
