import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        # Three average pooling layers with different pool sizes and strides
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)

        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)

        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)

        # Concatenate the flatten outputs
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Fully connected layer followed by reshaping
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)  # Assuming we want to reshape to (4, 4, 4)

    # Block 2
    def block_2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2: Two 3x3 Convolutions stacked after a 1x1 Convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: Single 3x3 Convolution after a 1x1 Convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4: Average Pooling with a 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenate outputs from all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block2_output = block_2(input_tensor=reshaped)

    # Flatten the output and add fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model