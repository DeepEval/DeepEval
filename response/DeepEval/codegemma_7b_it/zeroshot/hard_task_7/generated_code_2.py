from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Reshape, Permute, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv_init = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)

    # Block 1
    group_size = 32
    group_count = 2
    channels = 16
    split_axis = group_size * group_count
    shape_before_split = tf.shape(conv_init)
    x = Reshape((-1, split_axis, channels // group_count))(conv_init)
    x0 = Lambda(lambda x: x[:, :, :, 0:channels])(x)
    x1 = Lambda(lambda x: x[:, :, :, channels:channels*2])(x)
    x0 = Conv2D(filters=channels, kernel_size=(1, 1), padding='same', activation='relu')(x0)
    x1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x1)
    x1 = Conv2D(filters=channels, kernel_size=(1, 1), padding='same', activation='relu')(x1)
    x = Concatenate(axis=3)([x0, x1])
    x = Reshape((-1, shape_before_split[1], shape_before_split[2], channels))(x)

    # Block 2
    shape_before_reshape = tf.shape(x)
    x = Reshape((-1, shape_before_reshape[1] * shape_before_reshape[2], shape_before_reshape[3]))(x)
    x = Permute((2, 1, 3))(x)
    x = Reshape((-1, shape_before_reshape[3], shape_before_reshape[1], shape_before_reshape[2]))(x)

    # Final output
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Model definition
    model = Model(inputs, outputs)

    return model