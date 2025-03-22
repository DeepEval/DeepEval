import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout, Conv2D
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average Pooling and Concatenation
    def block_1(input_tensor):
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)

        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)

        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)

        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Block 2: Multi-path Convolutions
    def block_2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2: Stacked 1x1 and 3x3 Convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: 1x1 followed by a single 3x3 Convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4: Average Pooling followed by 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenate outputs of all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block_1(input_layer)
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(-1, 1, 1))(dense)  # Reshape to 4D for the second block

    block2_output = block_2(reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model