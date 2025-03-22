import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Concatenate, Conv2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        # Path 1: Average Pooling 1x1
        avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(avg_pool1)
        drop1 = Dropout(0.5)(flatten1)

        # Path 2: Average Pooling 2x2
        avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(avg_pool2)
        drop2 = Dropout(0.5)(flatten2)

        # Path 3: Average Pooling 4x4
        avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(avg_pool3)
        drop3 = Dropout(0.5)(flatten3)

        # Concatenate the outputs of the three paths
        output_tensor = Concatenate()([drop1, drop2, drop3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        # Branch 1: 1x1 Convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Branch 2: (1x1 Convolution followed by 3x3 Convolution)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: (1x1 Convolution followed by 3x3 Convolution, then another 3x3 Convolution)
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

        # Branch 4: (Average Pooling followed by 1x1 Convolution)
        branch4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate the outputs of the four branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    # Execute blocks
    block1_output = block_1(input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)  # reshape for block 2
    block2_output = block_2(reshaped)

    # Final classification layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model