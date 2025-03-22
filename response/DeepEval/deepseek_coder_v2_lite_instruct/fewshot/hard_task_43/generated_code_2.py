import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)

    def block_2(input_tensor):
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
        conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
        
        # Branch 3: Average pooling
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten = Flatten()(pool)
        
        # Concatenate outputs of all branches
        output_tensor = Concatenate()([conv2, conv6, flatten])
        return output_tensor

    block2_output = block_2(input_tensor=reshaped)

    # Final classification
    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model