import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

def depthwise_conv2d(input_tensor, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=None):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    kernel_shape = (kernel_size[0], kernel_size[1], input_tensor.shape[channel_axis], kernel_size[2])
    strides = (1, strides[0], strides[1], 1)
    depthwise_kernel = K.variable(tf.random.normal(kernel_shape, stddev=0.01))
    padding_ = padding if padding != 'valid' else 'same'
    x = layers.ZeroPadding2D(padding=padding_)(input_tensor)
    x = layers.DepthwiseConv2D(kernel_size=(kernel_size[0], kernel_size[1]), strides=strides, padding='valid', use_bias=False,
                                kernel_regularizer=kernel_regularizer)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    return x

def pointwise_conv2d(input_tensor, kernel_size, strides=(1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=None):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    kernel_shape = (kernel_size, kernel_size, input_tensor.shape[channel_axis], kernel_size)
    strides = (1, strides[0], strides[1], 1)
    pointwise_kernel = K.variable(tf.random.normal(kernel_shape, stddev=0.01))
    x = layers.Conv2D(kernel_size, kernel_size, strides=strides, padding=padding, use_bias=False,
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input_tensor)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    return x

def residual_block(input_tensor, kernel_size, filters, strides=(1, 1), dropout_rate=0.0):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    conv_shortcut = input_tensor
    x = depthwise_conv2d(input_tensor, kernel_size, strides, padding='same')
    x = pointwise_conv2d(x, 1, strides, padding='same')
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.add([x, conv_shortcut])
    return x

def classifier(input_shape, num_classes, block_fn, layers, filters, dropout_rate):
    img_input = layers.Input(shape=input_shape)

    x = img_input
    for i in range(layers):
        x = block_fn(x, kernel_size=(3, 3), filters=filters, strides=(1, 1), dropout_rate=dropout_rate)
        filters *= 2

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=img_input, outputs=output)
    return model

def dl_model():
    input_shape = (28, 28, 1)
    num_classes = 10
    layers = 3
    filters = 64
    dropout_rate = 0.2

    model = classifier(input_shape, num_classes, residual_block, layers, filters, dropout_rate)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model