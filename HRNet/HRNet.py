#!/usr/bin/env python3

# from base.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
import tensorflow.keras.layers as layers
import tensorflow.keras as keras


BN_MOMENTUM = 0.1


def conv3x3(out_planes, stride=1):
    return Sequential([
        layers.ZeroPadding2D(padding=(1, 1)),
        Conv2D(
            filters=out_planes,
            kernel_size=(3, 3),
            strides=(stride, stride),
        )
    ])


class BasicBlock(layers.Layer):

    expansion = 1

    def __init__(self, _in_channel, output_dim, stride=1, downsample=None, **kwargs):
        self.output_dim = output_dim
        self.downsample = downsample
        self.stride = stride

        super().__init__(**kwargs)

    def build(self, input_shape):
        stride = self.stride
        output_dim = self.output_dim
        self.conv1 = conv3x3(output_dim, stride)
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.relu = layers.ReLU()
        self.conv2 = conv3x3(output_dim)
        self.bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM)

    def call(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(layers.Layer):

    expansion = 4

    def __init__(self, _in_channel, output_dim, stride=1, downsample=None, **kwargs):
        self.output_dim = output_dim
        self.downsample = downsample
        self.stride = stride

        super().__init__(**kwargs)

    def build(self, input_shape):

        self.conv1 = layers.Conv2D(
            filters=self.output_dim,
            kernel_size=(1, 1),
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.conv2 = Sequential([
            layers.ZeroPadding2D(padding=(1, 1)),
            Conv2D(
                filters=self.output_dim,
                kernel_size=(3, 3),
                strides=(self.stride, self.stride),
                use_bias=False
            )

        ])
        self.bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM)

        self.conv3 = layers.Conv2D(
            filters=self.output_dim * self.expansion,
            kernel_size=(1, 1),
            use_bias=False
        )
        self.bn3 = layers.BatchNormalization(momentum=BN_MOMENTUM)

        self.relu = layers.ReLU()

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(layers.Layer):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True, **kwargs):

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.num_channels = num_channels

        super().__init__(**kwargs)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index]*block.expansion:
            downsample = Sequential([
                Conv2D(
                    num_channels[branch_index]*block.expansion,
                    kernel_size=(1, 1),
                    strides=(stride, stride),
                    use_bias=False
                ),
                BatchNormalization(momentum=BN_MOMENTUM)
            ])

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index]*block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return Sequential(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return branches

    def _make_fuse_layers(self):

        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        Sequential([
                            Conv2D(
                                num_inchannels[i],
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                use_bias=False
                            ),
                            BatchNormalization(),
                            layers.UpSampling2D(
                                size=(2**(j-i), 2**(j-i)),
                                interpolation='nearest'
                            )
                        ])
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i-j-1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(Sequential([
                                layers.ZeroPadding2D((1, 1,)),
                                Conv2D(
                                    num_outchannels_conv3x3,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    use_bias=False
                                ),
                                BatchNormalization()
                            ]))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(Sequential([
                                layers.ZeroPadding2D((1, 1,)),
                                Conv2D(
                                    num_outchannels_conv3x3,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    use_bias=False
                                ),
                                BatchNormalization(),
                                layers.ReLU()
                            ]))
                    fuse_layer.append(Sequential(conv3x3s))
            fuse_layers.append(fuse_layer)
        return fuse_layers

    def build(self, input_shape):

        self.branches = self._make_branches(
            self.num_branches,
            self.blocks,
            self.num_blocks,
            self.num_channels,
        )

        self.fuse_layers = self._make_fuse_layers()
        self.relu = layers.ReLU()

    def get_num_inchannels(self):
        return self.num_inchannels

    def call(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(keras.Model):
    def __init__(self, cfg):
        super(PoseHighResolutionNet, self).__init__(name='pose_hrnet')
        self.cfg = cfg
        self.inplanes = 64

        extra = cfg['MODEL']['EXTRA']

        self.conv1 = Sequential([
            layers.ZeroPadding2D(
                padding=(1, 1),
            ),
            Conv2D(
                64,
                kernel_size=(3, 3),
                strides=(2, 2),
                use_bias=False,
            )
        ])
        self.bn1 = BatchNormalization(momentum=BN_MOMENTUM)
        self.conv2 = Sequential([
            layers.ZeroPadding2D(padding=(1, 1)),
            Conv2D(
                64,
                kernel_size=(3, 3),
                strides=(2, 2),
                use_bias=False
            )
        ])
        self.bn2 = BatchNormalization()
        self.relu = layers.ReLU()
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            x * block.expansion
            for x in num_channels
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            x * block.expansion
            for x in num_channels
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            x * block.expansion
            for x in num_channels
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False
        )

        final_layer = []
        if extra['FINAL_CONV_KERNEL'] == 3:
            final_layer.append(layers.ZeroPadding2D(padding=(1, 1)))
        final_layer.append(
            Conv2D(
                cfg['MODEL']['NUM_JOINTS'],
                kernel_size=(extra['FINAL_CONV_KERNEL'],) * 2,
                strides=(1, 1),

            )
        )
        self.final_layer = Sequential(final_layer)

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(
        self,
        num_channels_pre_layer,
        num_channels_cur_layer,
    ):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(Sequential([
                        layers.ZeroPadding2D(padding=(1, 1)),
                        Conv2D(
                            num_channels_cur_layer[i],
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            use_bias=False
                        ),
                        BatchNormalization(),
                        layers.ReLU()
                    ]))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(Sequential([
                        layers.ZeroPadding2D(padding=(1, 1)),
                        Conv2D(
                            outchannels,
                            kernel_size=(3, 3,),
                            strides=(2, 2,),
                            use_bias=False
                        ),
                        BatchNormalization(),
                        layers.ReLU()
                    ]))
                transition_layers.append(Sequential(conv3x3s))

        return transition_layers

    def _make_stage(
        self,
        layer_config,
        num_inchannels,
        multi_scale_output=True
    ):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):

            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return modules, num_inchannels

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential([
                Conv2D(
                    planes*block.expansion,
                    kernel_size=(1, 1),
                    strides=(stride, stride),
                    use_bias=False
                ),
                BatchNormalization()
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self._forward_stage(self.stage2, x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self._forward_stage(self.stage3, x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self._forward_stage(self.stage4, x_list)
        x = self.final_layer(y_list[0])

        return x

    def _forward_stage(self, stage, xs):
        ys = xs

        for module in stage:
            ys = module(ys)
            if not isinstance(ys, list):
                ys = [ys]
        return ys


def get_pose_net(cfg, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    return model


if __name__ == '__main__':
    import yaml
    from keras import losses
    import tensorflow as tf
    import numpy as np
    cfg = yaml.load(
        open('cfg.yml'),
        Loader=yaml.FullLoader
    )
    model = get_pose_net(cfg)
    model.compile(
        'adam',
        loss=losses.mean_squared_error,
        metrics=['accuracy']
    )
    # model.fit(
    #     np.random.random([1, 256, 192, 3]),
    #     np.random.random([1, 256, 192, 20])

    # )
    model.predict(
        tf.zeros([1, 256, 192, 3]),
        batch_size=1
    )
    print(model.summary())