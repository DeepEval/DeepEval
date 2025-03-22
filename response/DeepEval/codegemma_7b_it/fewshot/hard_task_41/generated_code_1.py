import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Dropout

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        maxpool1_flatten = Flatten()(maxpool1)
        maxpool1_flatten = Dropout(0.3)(maxpool1_flatten)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        maxpool2_flatten = Flatten()(maxpool2)
        maxpool2_flatten = Dropout(0.3)(maxpool2_flatten)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        maxpool3_flatten = Flatten()(maxpool3)
        maxpool3_flatten = Dropout(0.3)(maxpool3_flatten)
        output_tensor = Concatenate()([maxpool1_flatten, maxpool2_flatten, maxpool3_flatten])
        return output_tensor

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model