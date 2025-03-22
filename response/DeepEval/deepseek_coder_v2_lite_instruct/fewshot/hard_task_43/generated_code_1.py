import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, MaxPooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Feature extraction using average pooling layers of different scales
    def block_1(input_tensor):
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Block 2: Classification using three branches
    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 2: 3x3 convolution
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 3: 1x7 convolution followed by 7x1 convolution
        conv3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        # Branch 4: 3x3 convolution with average pooling
        pool4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool4)
        output_tensor = Concatenate()([conv1, conv2, conv4, conv5])
        return output_tensor

    # Applying Block 1
    block1_output = block_1(input_tensor=input_layer)
    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense1)

    # Applying Block 2
    block2_output = block_2(input_tensor=reshaped)
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model