from models.model import create_model, load_model, save_model
from opts import opts

from models.large_hourglass import stacked_hourglass
from mxnet import nd
import mxnet as mx

print('Creating model...')
opt = opts().init()
print(opt.arch)
model = create_model(opt.arch, opt.heads, opt.head_conv)
model.collect_params().initialize()

#'''
X   = nd.random.uniform(shape=(1, 3, 256, 256))
print("\t Input shape: ", X.shape)
Y   = model(X)
print("\t output:", Y)
#'''

'''
model.hybridize()
model.collect_params().initialize()
x = mx.sym.var('data')
symbol = model(x)
mx.viz.plot_network(symbol)
'''

print("\n\nSaving model...")
save_model(model, "./init_params.params")



# call:
# python train.py ctdet --arch hourglass
