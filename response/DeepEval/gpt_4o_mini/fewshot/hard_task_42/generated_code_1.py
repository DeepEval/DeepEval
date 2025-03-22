import keras
import tensorflow as tf
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dropout, Dense, Reshape, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        flatten1 = Flatten()(path1)
        flatten2 = Flatten()(path2)
        flatten3 = Flatten()(path3)

        # Applying Dropout for regularization
        drop1 = Dropout(rate=0.5)(flatten1)
        drop2 = Dropout(rate=0.5)(flatten2)
        drop3 = Dropout(rate=0.5)(flatten3)

        output_tensor = Concatenate()([drop1, drop2, drop3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Fully connected layer and reshape
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)

    # Block 2
    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 1x1 followed by 1x7 and 7x1 convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 followed by alternating 7x1 and 1x7 convolutions
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)

        # Path 4: Average pooling followed by 1x1 convolution
        path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        # Concatenate outputs from all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block2_output = block_2(input_tensor=reshaped)

    # Final flatten and fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model