import d2lzh as d2l
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes+1), kernel_size=3, padding=1)

def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors*4, kernel_size=3, padding=1)

def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(nd.zeros((2,8,20,20)), cls_predictor(5,10))
Y2 = forward(nd.zeros((2,16,10,10)), cls_predictor(3,10))
print(Y1.shape, Y2.shape)

def flatten_pred(pred):
    return pred.transpose((0,2,3,1)).flatten()   # (b, c, w, h) -> (b, h, w, c) -> b * h * w * c

def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)

concated = concat_preds([Y1, Y2])
#print(concated.shape)
print(55*20*20 + 33*10*10)

def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        layers = []
        layers.append(nn.Conv2D(num_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm(in_channels=num_channels))
        layers.append(nn.Activation('relu'))

        blk.add(*layers)
    blk.add(nn.MaxPool2D(2))
    return  blk

def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

# downsample to 1/8, with 64 output channels
temp_input = nd.zeros((2,3,256,256))
temp_output = forward(temp_input, base_net())
print(temp_output.shape)


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
print(num_anchors)


class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # setattr: self.blk_i = get_blk(i)
            setattr(self, 'blk_{}'.format(i), get_blk(i))
            setattr(self, 'cls_{}'.format(i), cls_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_{}'.format(i), bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None]*5, [None]*5, [None]*5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, 
                                                                     getattr(self, 'blk_{}'.format(i)), 
                                                                     sizes[i], 
                                                                     ratios[i], 
                                                                     getattr(self, 'cls_{}'.format(i)), 
                                                                     getattr(self, 'bbox_{}'.format(i)), 
                                                                     )
            print("cls_preds[{}].shape = {}".format(i, cls_preds[i].shape))
            print("bbox_preds[{}].shape = {}".format(i, bbox_preds[i].shape))
        return ( nd.concat(*anchors, dim=1),
                 concat_preds(cls_preds).reshape((0, -1, self.num_classes+1)),  # 0 means unchanged
                 concat_preds(bbox_preds)
        )

net = TinySSD(num_classes=1)
net.initialize()
X = nd.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)


batch_size = 32
train_iter, _ = d2l.load_data_pikachu(batch_size)

ctx, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init = init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2, 'wd': 5e-4})  # wd: weight decay

cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    l_cls = cls_loss(cls_preds, cls_labels)
    l_bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return l_cls + l_bbox

def cls_eval(cls_preds, cls_labels):
    return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()


for epoch in range(20):
    acc_sum, mae_sum, n, m = 0, 0, 0
    train_iter.reset()
    start = time.time()

    for batch in train_iter:
        X = batch.data[0].as_in_context(ctx)
        Y = batch.label[0].as_in_context(ctx)

        with autograd.record():
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(anchors, Y, cls_preds.transpose((0, 2, 1)))
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        
        l.backward()
        trainer.step(batch_size)

        acc_sum += cls_eval(cls_preds, cls_labels)
        n += cls_labels.size
        mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
        m += bbox_labels.size
