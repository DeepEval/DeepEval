import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], shape[3] // 3, 3))(input_tensor)
        transposed = tf.transpose(reshaped, [0, 3, 1, 2])
        reshaped_again = Reshape(target_shape=(shape[1], shape[2], 3, shape[3] // 3))(transposed)
        shuffled = tf.transpose(reshaped_again, [0, 2, 3, 1])
        return shuffled

    def block_3(input_tensor):
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    main_path_output = block_1(input_tensor=input_layer)
    main_path_output = block_2(input_tensor=main_path_output)
    main_path_output = block_3(input_tensor=main_path_output)
    main_path_output = block_1(input_tensor=main_path_output)

    branch_path_output = MaxPooling2D(pool_size=(32, 32), strides=1, padding='valid')(input_layer)

    combined_output = Concatenate()([main_path_output, branch_path_output])

    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model