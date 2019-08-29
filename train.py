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
    val_metric = COCODetectionMetric(val_dataset,
                                     '_eval',
                                     cleanup=True,
                                     data_shape=(args.data_shape, args.data_shape))
    # coco validation is slow, consider increase the validation interval
    args.val_interval = 10
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx):
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


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)

    return




# call:
# python train.py ctdet --arch hourglass
if __name__ == "__main__":
    """ 1. network """

    print('Creating model...')
    opt = opts().init()
    print(opt.arch)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.collect_params().initialize(init=init.Xavier())

    X   = nd.random.uniform(shape=(1, 3, 256, 256))
    print("\t Input shape: ", X.shape)
    Y   = model(X)
    print("\t output:", Y)

    param = model.collect_params()
    param_keys = param.keys()
    param_keys_residual_1 = [param[param_key] for param_key in param_keys if "hourglassnet0_residual1_conv1_weight" in param_key]
    print(param_keys_residual_1)

    print("\n\nSaving model...")
    save_model(model, "./init_params.params")


    """ 2. Dataset """



    """ 3. Training """
