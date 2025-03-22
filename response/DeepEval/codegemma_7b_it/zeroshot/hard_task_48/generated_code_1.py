from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Activation, MaxPooling2D, concatenate, Flatten, Dense

def dw_conv_block(x, kernel_size, strides, padding='same', use_bn=True):
    """ Depthwise convolution block."""
    if strides > 1:
        x = Conv2D(kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, groups=kernel_size)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, groups=kernel_size)(x)

    return x


def pw_conv_block(x, filters, strides, padding='same', use_bn=True):
    """ Pointwise convolution block."""
    if strides > 1:
        x = Conv2D(filters=filters, kernel_size=1, strides=strides, padding=padding, use_bias=False)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=1, strides=strides, padding=padding, use_bias=False)(x)

    return x


def block_1(x):
    """ Block 1."""
    x_1 = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=3)) (x)
    x_1_1 = dw_conv_block(x_1[0], kernel_size=1, strides=1)
    x_1_2 = dw_conv_block(x_1[1], kernel_size=3, strides=1)
    x_1_3 = dw_conv_block(x_1[2], kernel_size=5, strides=1)
    x_1 = concatenate([x_1_1, x_1_2, x_1_3])

    x_2 = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=3)) (x)
    x_2_1 = dw_conv_block(x_2[0], kernel_size=1, strides=1)
    x_2_2 = dw_conv_block(x_2[1], kernel_size=3, strides=1)
    x_2_3 = dw_conv_block(x_2[2], kernel_size=5, strides=1)
    x_2 = concatenate([x_2_1, x_2_2, x_2_3])

    x_3 = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=3)) (x)
    x_3_1 = dw_conv_block(x_3[0], kernel_size=1, strides=1)
    x_3_2 = dw_conv_block(x_3[1], kernel_size=3, strides=1)
    x_3_3 = dw_conv_block(x_3[2], kernel_size=5, strides=1)
    x_3 = concatenate([x_3_1, x_3_2, x_3_3])

    x = concatenate([x_1, x_2, x_3])
    x = dw_conv_block(x, kernel_size=1, strides=1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def block_2(x):
    """ Block 2."""
    path_1 = pw_conv_block(x, filters=64, strides=1)

    path_2 = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    path_2 = pw_conv_block(path_2, filters=64, strides=1)

    path_3 = pw_conv_block(x, filters=64, strides=1)
    path_3 = dw_conv_block(path_3, kernel_size=1, strides=1)
    path_3_1 = dw_conv_block(path_3, kernel_size=3, strides=1, padding='same')
    path_3_2 = dw_conv_block(path_3, kernel_size=3, strides=1, padding='same')
    path_3 = concatenate([path_3_1, path_3_2])

    path_4 = pw_conv_block(x, filters=64, strides=1)
    path_4 = dw_conv_block(path_4, kernel_size=1, strides=1)
    path_4_1 = dw_conv_block(path_4, kernel_size=3, strides=1, padding='same')
    path_4_2 = dw_conv_block(path_4, kernel_size=3, strides=1, padding='same')
    path_4 = concatenate([path_4_1, path_4_2])

    x = concatenate([path_1, path_2, path_3, path_4])

    x = dw_conv_block(x, kernel_size=1, strides=1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def dl_model():
    """ Construct model."""
    input_img = Input(shape=(32, 32, 3))

    # Block 1
    x = block_1(input_img)

    # Block 2
    x = block_2(x)

    # Classification layer
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_img, outputs=x)

    return model