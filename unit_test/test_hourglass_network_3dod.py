import sys
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/")

from models.model import create_model, load_model, save_model
from opts import opts

from models.hourglass import stacked_hourglass
from mxnet import nd, gluon, init
import mxnet as mx

print('Creating model...')
opt = opts().init()
print(opt.arch)
ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
ctx = ctx if ctx else [mx.cpu()]
model = create_model(opt.arch, opt.heads, opt.head_conv, ctx)
model.collect_params().initialize(init=init.Xavier())

#decode_centernet_3dod(heat, rot, depth, dim, wh=None, reg=None, K=40)
#opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
X   = nd.random.uniform(shape=(16, 3, 512, 512))
print("\t Input shape: ", X.shape)
Y   = model(X)
print(Y[0].keys())
print("output: center", Y[0]["hm"].shape)

print("output: 2d bbox wh", Y[0]["wh"].shape)
print("output: 2d bbox offset", Y[0]["reg"].shape)

print("output: 3d rotation", Y[0]["rot"].shape)
print("output: 3d depth", Y[0]["dep"].shape)
print("output: 3d dimention", Y[0]["dim"].shape)

# call:
# python test_hourglass_network_3dod.py --task ddd --arch hourglass
