import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def split_channels(inputs):
    shape = tf.shape(inputs)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]

    x1 = inputs[:, :, :, 0:channels//3]
    x2 = inputs[:, :, :, channels//3:2*channels//3]
    x3 = inputs[:, :, :, 2*channels//3:channels]

    return x1, x2, x3

def multi_scale_conv(inputs, kernel_size, filters):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(inputs)
    return conv

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split channels
    x1, x2, x3 = Lambda(split_channels)(input_layer)

    # Multi-scale convolutional layers
    conv1_1x1 = multi_scale_conv(x1, kernel_size=(1, 1), filters=64)
    conv1_3x3 = multi_scale_conv(x1, kernel_size=(3, 3), filters=64)
    conv1_5x5 = multi_scale_conv(x1, kernel_size=(5, 5), filters=64)

    conv2_1x1 = multi_scale_conv(x2, kernel_size=(1, 1), filters=64)
    conv2_3x3 = multi_scale_conv(x2, kernel_size=(3, 3), filters=64)
    conv2_5x5 = multi_scale_conv(x2, kernel_size=(5, 5), filters=64)

    conv3_1x1 = multi_scale_conv(x3, kernel_size=(1, 1), filters=64)
    conv3_3x3 = multi_scale_conv(x3, kernel_size=(3, 3), filters=64)
    conv3_5x5 = multi_scale_conv(x3, kernel_size=(5, 5), filters=64)

    # Concatenate outputs
    concat = Concatenate()([conv1_1x1, conv1_3x3, conv1_5x5,
                           conv2_1x1, conv2_3x3, conv2_5x5,
                           conv3_1x1, conv3_3x3, conv3_5x5])

    # Batch normalization
    batch_norm = BatchNormalization()(concat)

    # Flatten
    flatten = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model