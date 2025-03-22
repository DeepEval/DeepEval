import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    def separable_conv_block(input_tensor, filters, kernel_size):
        x = Conv2D(filters, kernel_size, padding='same', activation='relu', depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(input_tensor)
        x = BatchNormalization()(x)
        return x

    # Feature extraction for each channel group
    conv1x1 = separable_conv_block(split_layer[0], filters=32, kernel_size=(1, 1))
    conv3x3 = separable_conv_block(split_layer[1], filters=32, kernel_size=(3, 3))
    conv5x5 = separable_conv_block(split_layer[2], filters=32, kernel_size=(5, 5))

    # Concatenate the outputs from these three groups
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Pass the flattened output through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model