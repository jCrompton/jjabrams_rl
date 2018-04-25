import numpy as np
import keras
import string

from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, SeparableConv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D

from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K

class Blocks:
    def __init__(self, separable=True, activation='relu'):
        print('Initialized the building blocks module')
        self.conv = SeparableConv2D if separable else Conv2D
        self.activation = activation

    def residual_block(self, input_tensor, kernel_size, filters, number_of_conv_blocks, number_of_id_blocks, stage, strides=(2, 2)):
        conv_block = input_tensor
        for i in range(number_of_conv_blocks):
            block_name = '{}_CONV{}'.format(stage, string.ascii_lowercase[i%26])
            conv_block = self.conv_block(conv_block, kernel_size, filters, i, block_name, strides=strides)
        id_block = conv_block
        for j in range(number_of_id_blocks):
            block_name = '{}_ID{}'.format(stage, string.ascii_lowercase[j%26])
            id_block = self.identity_block(id_block, kernel_size, filters, j, block_name)
        return id_block

    def identity_block_T(self, input_tensor, kernel_size, filters, stage, block):
        pass

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        # Returns
        Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = self.conv(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation(self.activation)(x)

        x = self.conv(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation(self.activation)(x)

        x = self.conv(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation(self.activation)(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        # Returns
        Output tensor for the block.
        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = self.conv(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)

        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation(self.activation)(x)
        
        x = self.conv(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation(self.activation)(x)

        x = self.conv(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
        shortcut = self.conv(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation(self.activation)(x)
        return x
