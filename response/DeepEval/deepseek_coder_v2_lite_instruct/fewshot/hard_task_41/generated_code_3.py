import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Dropout, Lambda, Reshape
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(avgpool1)
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(avgpool2)
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(avgpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        output_tensor = Dropout(0.5)(output_tensor)  # Regularization using dropout
        return output_tensor

    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 2: <1x1 convolution
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
        # Branch 3: 3x3 convolution
        conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 4: <average pooling, 1x1 convolution>
        avgpool4_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv4_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avgpool4_1)

        output_tensor = Concatenate()([conv1_1, conv2_2, conv3_1, conv4_2])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model