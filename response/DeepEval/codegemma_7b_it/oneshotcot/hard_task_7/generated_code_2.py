import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    # Block 1
    def block1(input_tensor):
        # Split input into two groups
        group1, group2 = tf.split(input_tensor, num_or_size_splits=2, axis=-1)

        # Operations for group 1
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_uniform')(path1)
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)

        # Operations for group 2
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)

        # Concatenate outputs from both groups
        output_tensor = Concatenate()([path1, path2])
        return output_tensor

    # Block 2
    def block2(input_tensor):
        # Get input shape
        input_shape = keras.backend.int_shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        # Reshape input into four groups
        groups = 2
        groups_channels = channels // groups
        input_reshaped = tf.reshape(input_tensor, shape=(-1, height, width, groups, groups_channels))

        # Swap third and fourth dimensions
        input_reshaped = tf.transpose(input_reshaped, perm=[0, 1, 3, 2, 4])

        # Reshape back to original shape
        input_reshaped = tf.reshape(input_reshaped, shape=(-1, height, width, channels))

        # Flatten and pass through fully connected layer
        output_tensor = Flatten()(input_reshaped)
        output_tensor = Dense(units=10, activation='softmax')(output_tensor)
        return output_tensor

    # Combine blocks
    block1_output = block1(input_tensor=conv)
    block2_output = block2(input_tensor=block1_output)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model