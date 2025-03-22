import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate, Lambda, DepthwiseConv2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First block
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    # Flatten and concatenate pooling results
    flatten_1x1 = Flatten()(avg_pool_1x1)
    flatten_2x2 = Flatten()(avg_pool_2x2)
    flatten_4x4 = Flatten()(avg_pool_4x4)
    concat_pools = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])

    # Fully connected layer and reshape
    dense_block = Dense(units=128, activation='relu')(concat_pools)
    reshape_layer = tf.reshape(dense_block, shape=(-1, 4, 4, 1))

    # Second block
    def depthwise_block(input_tensor, kernel_size):
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=(1, 1), padding='same')(input_tensor)
        return conv

    # Process input into four groups
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape_layer)

    # Apply depthwise separable convolutional layers
    conv_1x1 = Lambda(lambda x: depthwise_block(x, kernel_size=(1, 1)))(split_input[0])
    conv_3x3 = Lambda(lambda x: depthwise_block(x, kernel_size=(3, 3)))(split_input[1])
    conv_5x5 = Lambda(lambda x: depthwise_block(x, kernel_size=(5, 5)))(split_input[2])
    conv_7x7 = Lambda(lambda x: depthwise_block(x, kernel_size=(7, 7)))(split_input[3])

    # Concatenate outputs and flatten
    concat_conv = Concatenate()([conv_1x1, conv_3x3, conv_5x5, conv_7x7])
    flatten_conv = Flatten()(concat_conv)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_conv)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model