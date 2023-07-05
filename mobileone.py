from typing import Optional, List, Tuple
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

__all__ = ['MobileOneTF', 'mobileoneTF', 'reparameterize_modelTF']

class MobileOneBlockTF(layers.Layer):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlockTF, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)#TODO
        else:
            self.se = layers.Layer()#Identity
        self.activation = layers.LeakyReLU()#layers.ReLU()

        if inference_mode:
            self.reparam_conv = layers.Conv2D(filters=out_channels,
                                             kernel_size=kernel_size,
                                             strides=stride,
                                             padding='same',
                                             groups=groups,
                                             use_bias=True)
         
        else:                
            # Re-parameterizable skip connection
            self.rbr_skip =  layers.BatchNormalization() \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                             padding=padding))
            self.rbr_conv = rbr_conv

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                              padding=0)

    def call(self, x):
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        conv = self.rbr_conv.layers[0].layers[0]
        
        if conv.groups > 1 and conv.groups == conv.filters:#group conv -> dwconv
            self.reparam_conv = layers.DepthwiseConv2D(
                                         kernel_size=conv.kernel_size,
                                         strides=conv.strides,
                                         padding='same',
                                         use_bias=True,
                                         weights=[np.transpose(kernel.numpy(),[0,1,3,2]), bias.numpy()])
        else:
            self.reparam_conv = layers.Conv2D(filters=conv.filters,
                                         kernel_size=conv.kernel_size,
                                         strides=conv.strides,
                                         padding='same',
                                         groups=conv.groups,
                                         use_bias=True,
                                         weights=[kernel.numpy(), bias.numpy()])
        
        # Delete un-used branches
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    #tf kernel:[filter_height, filter_width, in_channels, out_channels]
    #pytorch kernel:[out_channels, in_channels, filter_height, filter_width]
    def _get_kernel_bias(self):
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            if self.kernel_size == 3 and self.stride == 2:
                kernel_scale = tf.pad(kernel_scale, tf.constant([[0, 2], [0, 2], [0, 0], [0, 0]]), "CONSTANT")#stride=2, cov1x1/conv3x3 center isn't align
            else:
                kernel_scale = tf.pad(kernel_scale, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), "CONSTANT")
            

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
            

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv.layers[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, Sequential):
            kernel = branch.layers[0].kernel #layers[0]=conv
            running_mean = branch.layers[1].moving_mean #layers[1]=bn
            running_var = branch.layers[1].moving_variance
            gamma = branch.layers[1].gamma
            beta = branch.layers[1].beta
            eps = branch.layers[1].epsilon
            
        else:
            assert isinstance(branch, layers.BatchNormalization)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.kernel_size, 
                                         self.kernel_size, 
                                         input_dim, 
                                         self.in_channels), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[self.kernel_size // 2, 
                                 self.kernel_size // 2,
                                 i % input_dim,
                                 i] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
            
        std = tf.sqrt((running_var + eps))
        t = tf.reshape((gamma / std), (1, 1, 1, -1))
        #t = (gamma / std).reshape(-1, 1, 1, 1)   
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = Sequential()
        mod_list.add(layers.Conv2D(filters=self.out_channels,
                                     kernel_size=kernel_size,
                                     strides=self.stride,
                                     padding='same',
                                     groups=self.groups,
                                     use_bias=False))
        mod_list.add(layers.BatchNormalization())
        return mod_list



class MobileOneTF(tf.keras.Model):
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlockTF(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.linear = tf.keras.layers.Dense(num_classes)

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(MobileOneBlockTF(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlockTF(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return Sequential(blocks)

    def call(self, x):
        """ Apply forward pass. """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.linear(x)
        return x


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0),
           "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0),
           "use_se": True},
}


def mobileoneTF(num_classes: int = 1000, inference_mode: bool = False,
              variant: str = "s0"):
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model. """
    variant_params = PARAMS[variant]
    return MobileOneTF(num_classes=num_classes, inference_mode=inference_mode,
                     **variant_params)


def reparameterize_modelTF(model):
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    #model = copy.deepcopy(model) #clone model outside
    for layer in model.layers:
        if isinstance(layer, MobileOneBlockTF):
            layer.reparameterize()
        elif isinstance(layer, Sequential):
            reparameterize_modelTF(layer)
    return model