import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras import initializers
from keras.regularizers import l2
from tensorflow.keras import backend as K

def depthwise_conv2d(inputs, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)):
    return DepthwiseConv2D(kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inputs)

def pointwise_conv2d(inputs, filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)):
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inputs)

def block_1(inputs):
    path_1 = Lambda(lambda x: K.split(x, num_or_size_splits=3, axis=-1))(inputs)
    path_1 = Lambda(lambda x: [pointwise_conv2d(i, 32, (1, 1), padding='same', kernel_initializer='he_normal') for i in x])(path_1)
    path_1 = Lambda(lambda x: [depthwise_conv2d(i, (3, 3), padding='same', kernel_initializer='he_normal') for i in x])(path_1)
    path_1 = Lambda(lambda x: [pointwise_conv2d(i, 128, (1, 1), padding='same', kernel_initializer='he_normal') for i in x])(path_1)
    path_1 = Lambda(lambda x: [BatchNormalization()(i) for i in x])(path_1)
    path_1 = Lambda(lambda x: [K.concatenate(i) for i in x])(path_1)
    return path_1

def block_2(inputs):
    path_1 = pointwise_conv2d(inputs, 128, (1, 1), padding='same', kernel_initializer='he_normal')

    path_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputs)
    path_2 = pointwise_conv2d(path_2, 128, (1, 1), padding='same', kernel_initializer='he_normal')

    path_3 = pointwise_conv2d(inputs, 128, (1, 1), padding='same', kernel_initializer='he_normal')
    path_3 = depthwise_conv2d(path_3, (1, 3), padding='same', kernel_initializer='he_normal')
    path_3 = depthwise_conv2d(path_3, (3, 1), padding='same', kernel_initializer='he_normal')
    path_3 = pointwise_conv2d(path_3, 128, (1, 1), padding='same', kernel_initializer='he_normal')

    path_4 = pointwise_conv2d(inputs, 128, (1, 1), padding='same', kernel_initializer='he_normal')
    path_4 = depthwise_conv2d(path_4, (3, 3), padding='same', kernel_initializer='he_normal')
    path_4 = pointwise_conv2d(path_4, 128, (1, 1), padding='same', kernel_initializer='he_normal')
    path_4 = depthwise_conv2d(path_4, (1, 3), padding='same', kernel_initializer='he_normal')
    path_4 = depthwise_conv2d(path_4, (3, 1), padding='same', kernel_initializer='he_normal')
    path_4 = pointwise_conv2d(path_4, 128, (1, 1), padding='same', kernel_initializer='he_normal')
    path_4 = BatchNormalization()(path_4)

    outputs = Concatenate()([path_1, path_2, path_3, path_4])
    return outputs

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    block_1_output = block_1(inputs=input_layer)
    block_2_output = block_2(inputs=block_1_output)

    flatten_layer = Flatten()(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model