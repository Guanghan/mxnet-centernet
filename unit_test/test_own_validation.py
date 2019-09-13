'''
Author: Guanghan Ning
Date:   August, 2019
'''

import sys, os
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/")
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/dataset")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/dataset")

import mxnet as mx
from mxnet import nd, gluon, init, autograd

from opts import opts
from models.model import create_model, load_model, save_model

from coco_centernet import CenterCOCODataset
from detectors.center_detector import CenterDetector

from progress.bar import Bar
from utils.misc import AverageMeter

def get_coco_val(opt, coco_path="/export/guanghan/coco"):
    """Get coco dataset."""
    val_dataset = CenterCOCODataset(opt, split = 'val')   # custom dataset
    return val_dataset

def validate(model, dataset, opt, ctx):
    """Test on validation dataset."""
    detector = CenterDetector(opt)
    detector.model = model

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    print("Reporting every 1000 images...")
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        ret = detector.run(img_path)
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
    opt = opts().init()
    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    print("Using Devices: ", ctx)

    """ 1. network """
    print('Creating model...')
    print("Using network architecture: ", opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv, ctx = ctx)
    model = load_model(model, opt.load_model_path, ctx = ctx)

    """ 2. Dataset """
    val_dataset = get_coco_val(opt, "./data/coco")

    """ 3. Testing Validation """
    validate(model, val_dataset, opt, ctx)
