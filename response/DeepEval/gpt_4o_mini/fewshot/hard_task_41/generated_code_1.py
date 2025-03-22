import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        # Parallel paths with different pooling scales
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        dropout1 = Dropout(0.3)(flatten1)

        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        dropout2 = Dropout(0.3)(flatten2)

        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        dropout3 = Dropout(0.3)(flatten3)

        # Concatenate the outputs from the three paths
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Fully connected layer followed by reshaping
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)

    # Block 2
    def block_2(input_tensor):
        # Multiple branches for feature extraction
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

        branch4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate the outputs from all branches
        output_tensor = Concatenate()([branch2, branch3, branch4])
        return output_tensor

    block2_output = block_2(input_tensor=reshaped)

    # Final classification layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model