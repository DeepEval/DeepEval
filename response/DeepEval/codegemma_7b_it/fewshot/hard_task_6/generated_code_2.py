import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Concatenate, Reshape, DepthwiseConv2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        reshaped = Reshape((shape[1], shape[2], 3, shape[3]//3))(input_tensor)
        swapped = tf.transpose(reshaped, perm=[0, 1, 3, 2])
        reshaped_back = Reshape((shape[1], shape[2], shape[3]))(swapped)
        return reshaped_back

    def block_3(input_tensor):
        conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    branch_path = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    main_path = block_1(input_tensor=input_layer)
    main_path = block_1(input_tensor=main_path)
    main_path = block_2(input_tensor=main_path)
    main_path = block_3(input_tensor=main_path)

    concat = Concatenate()([branch_path, main_path])
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model