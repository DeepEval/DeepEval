import keras
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, AveragePooling2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs_groups[2])
        bn1 = BatchNormalization()(conv1)
        bn2 = BatchNormalization()(conv2)
        bn3 = BatchNormalization()(conv3)
        output_tensor = Concatenate()([bn1, bn2, bn3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    def block_2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        branch2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch2)
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        branch3_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same')(branch3)
        branch3_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same')(branch3)
        branch3 = Concatenate()([branch3_1, branch3_2])
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        branch4_1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same')(branch4)
        branch4_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same')(branch4)
        branch4 = Concatenate()([branch4_1, branch4_2])
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model