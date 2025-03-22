import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    def depthwise_separable_conv(input_tensor, kernel_size):
        # Depthwise convolution
        depthwise_conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_initializer='he_normal', activation='relu')(input_tensor)
        # Pointwise convolution
        pointwise_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        return pointwise_conv

    # Feature extraction for each group
    group1 = depthwise_separable_conv(split_layer[0], kernel_size=(1, 1))
    group2 = depthwise_separable_conv(split_layer[1], kernel_size=(3, 3))
    group3 = depthwise_separable_conv(split_layer[2], kernel_size=(5, 5))

    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([group1, group2, group3])

    # Flatten the fused features
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model