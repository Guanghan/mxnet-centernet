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
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx))
    anchors = anchors.as_in_context(mx.cpu())
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader






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



# call:
# python train.py ctdet --arch hourglass
