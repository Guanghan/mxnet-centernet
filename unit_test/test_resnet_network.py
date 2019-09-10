import sys
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/")

from models.model import create_model, load_model, save_model
from opts import opts

from models.resnet import get_pose_net
from mxnet import nd, gluon, init
import mxnet as mx

print('Creating model...')
opt = opts().init()
print(opt.arch)
model = get_pose_net(opt.heads, opt.head_conv, num_layers=18, load_pretrained =True)

#model.collect_params().initialize(init=init.Xavier())

X   = nd.random.uniform(shape=(16, 3, 512, 512))
print("\t Input shape: ", X.shape)
Y   = model(X)
print("output: heatmaps", Y[0]["hm"].shape)
print("output: wh_scale", Y[0]["wh"].shape)
print("output: xy_offset", Y[0]["reg"].shape)

param = model.collect_params()
param_keys = param.keys()
print(param_keys)
#param_keys_residual_1 = [param[param_key] for param_key in param_keys if "hourglassnet0_residual1_conv1_weight" in param_key]
#print(param_keys_residual_1)

'''
flag_save_model = False
if flag_save_model is True:
    print("\n\nSaving model...")
    save_model(model, "./init_params.params")
'''


# call:
# python test_resnet_network.py --arch res_18
