# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument,missing-docstring,too-many-lines
"""ResNets, implemented in Gluon."""
from __future__ import division

import os
import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.cnn.conv_layers import DeformableConvolution

#import sys, os
#sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/")
#from models.model import create_model, load_model, save_model

BN_MOMENTUM = 0.1


def download_pretrained_model():
    import gluoncv
    net = gluoncv.model_zoo.get_model("resnet18_v1", pretrained=True)
    return

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

def fill_up_weights(up, ctx = cpu()):
    w = up.weight.data()
    f = math.ceil(w.shape[2] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.shape[0]):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers, single_layer= False, ctx = cpu()):
    if single_layer:
        m = layers
        fill_fc_weights_single_layer(m, ctx)
    else:
        for m in layers:
            fill_fc_weights_single_layer(m, ctx)


def fill_fc_weights_single_layer(m, ctx = cpu()):
    if isinstance(m, nn.Conv2D):
        params = m.params
        for param_name in params.keys():
            param = params[param_name]
            if "weight" in param_name:
                param.initialize(init = mx.init.Normal(sigma=0.001), ctx = ctx)
            elif "bias" in param_name:
                param.initialize(init = mx.init.Constant(0), ctx = ctx)

# Blocks
class BasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels * 4, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x


class BottleneckV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels * 4, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


# Nets
class ResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class PoseResNet(HybridBlock):
    def __init__(self, block, layers, heads, head_conv, channels,
                 version = 1, ctx=cpu(), root='~/.mxnet/models', use_se=False, **kwargs):
        super(PoseResNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            thumbnail = False
            norm_layer = BatchNorm
            norm_kwargs = None
            last_gamma = False
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))

        # used for deconv layers
        self.deconv_with_bias = False
        self.inplanes = 512 #64
        self.heads = heads

        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.HybridSequential()
                fc.add(nn.Conv2D(head_conv, kernel_size=3, in_channels = 64, padding=1, use_bias=True),
                       nn.LeakyReLU(0),
                       nn.Conv2D(classes, kernel_size=1, in_channels = head_conv, strides=1, padding=0, use_bias=True))
                '''
                if 'hm' in head:
                    fc[-1].bias.set_data(-2.19)
                else:
                    fill_fc_weights(fc)
                '''
            else:
                fc = nn.Conv2D(classes,kernel_size=1, in_channels=64, strides=1, padding=0, use_bias=True)
                '''
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc, single_layer=True)
                '''
            self.__setattr__(head, fc)


    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer


    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]

            print("Deconv {}, fc, in_channels: {}, out_channels: {}".format(i, self.inplanes, planes))

            fc = DeformableConvolution(planes,
                     kernel_size=3, strides=1, num_deformable_group=1,
                     in_channels = self.inplanes,
                     padding=1, dilation=1, use_bias=False) # http://34.201.8.176/versions/1.5.0/_modules/mxnet/gluon/contrib/cnn/conv_layers.html
            '''
            fc = nn.Conv2D(planes,
                     kernel_size=3, strides=1,
                     in_channels = self.inplanes,
                     padding=1, dilation=1, use_bias=False)
            '''

            '''
            fill_fc_weights(fc, single_layer=True)
            '''


            print("Deconv {}, up, in_channels: {}, out_channels: {}".format(i, planes, planes))

            up = nn.Conv2DTranspose(
                    channels=planes,
                    kernel_size=kernel,
                    in_channels=planes,
                    strides=2,
                    padding=padding,
                    output_padding=output_padding,
                    use_bias=self.deconv_with_bias)
            '''
            fill_up_weights(up)
            '''

            layers.append(fc)
            layers.append(nn.BatchNorm(momentum=BN_MOMENTUM))
            layers.append(nn.LeakyReLU(0))
            layers.append(up)
            layers.append(nn.BatchNorm(momentum=BN_MOMENTUM))
            layers.append(nn.LeakyReLU(0))
            self.inplanes = planes

        deconv_layers = nn.HybridSequential()
        deconv_layers.add(*layers)
        return deconv_layers

    def hybrid_forward(self, F, x):
        x = self.features(x)
        #print(x.shape)
        x = self.deconv_layers(x)
        #print(x.shape)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattribute__(head)(x)
        return [ret]


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1}]


def get_pose_net(num_layers, heads, head_conv=256, load_pretrained = False, ctx = cpu()):
    block_type, layers, channels = resnet_spec[num_layers]

    version = 1  # Now I only implement resnet_v1
    block_class = resnet_block_versions[version-1][block_type]

    model = PoseResNet(block_class, layers, heads, head_conv, channels,
                       version = 1, ctx=cpu(), root='~/.mxnet/models', use_se=False)

    initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    model.collect_params().initialize(init= initializer, ctx = ctx)

    if num_layers == 18 and load_pretrained:
        #model_load_path = '/Users/guanghan.ning/.mxnet/models/resnet18_v1-a0666292.params'
        model_load_path = '/root/.mxnet/models/resnet18_v1-a0666292.params'
        model = load_model(model, model_load_path, ctx = ctx)
    return model


# Constructor
def get_resnet(version, num_layers, pretrained=False, ctx=cpu(),
               root='~/.mxnet/models', use_se=False, **kwargs):
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        if not use_se:
            net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx)
        else:
            net.load_parameters(get_model_file('se_resnet%d_v%d'%(num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net

def resnet18_v1(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_resnet(1, 18, use_se=False, **kwargs)
