import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Conv2D, Dropout, Flatten, Concatenate, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block: Pooling and concatenation
    def block_1(input_tensor):
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        concatenated = Concatenate()([flatten1, flatten2, flatten3])
        return concatenated

    # Second Block: Multi-scale feature extraction
    def block_2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2: 1x1 followed by two 3x3 Convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: 1x1 followed by 3x3 Convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4: Average Pooling followed by 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenating paths
        concatenated = Concatenate(axis=-1)([path1, path2, path3, path4])
        return concatenated

    # Apply first block
    block1_output = block_1(input_layer)
    dense_block1 = Dense(units=128, activation='relu')(block1_output)
    reshaped_block1 = Reshape(target_shape=(7, 7, 4))(dense_block1)

    # Apply second block
    block2_output = block_2(reshaped_block1)

    # Final fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model