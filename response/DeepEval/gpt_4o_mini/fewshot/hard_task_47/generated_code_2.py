import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Depthwise Separable Convolutions
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        norm1 = BatchNormalization()(conv1)

        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        norm2 = BatchNormalization()(conv2)

        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        norm3 = BatchNormalization()(conv3)

        output_tensor = Concatenate()([norm1, norm2, norm3])
        return output_tensor

    block1_output = block_1(input_layer)

    # Block 2: Multiple Branches
    def block_2(input_tensor):
        branch1 = Concatenate()([
            Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        ])

        branch2 = Concatenate()([
            Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        ])

        branch3 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)

        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    block2_output = block_2(block1_output)

    # Fully Connected Layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model