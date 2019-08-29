from models.model import create_model, load_model, save_model
from opts import opts

from models.large_hourglass import stacked_hourglass
from mxnet import nd, gluon, init
import mxnet as mx

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
