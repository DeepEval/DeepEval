import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        groups = 3
        channels = keras.backend.int_shape(input_tensor)[-1]
        channels_per_group = channels // groups

        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=groups, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=channels_per_group, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=channels_per_group, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=channels_per_group, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        input_shape = (shape[1], shape[2], shape[3] // 3, 3)

        group_shape = keras.backend.int_shape(input_tensor)
        group_shape = (group_shape[1], group_shape[2], 3, group_shape[3] // 3)

        input_reshaped = Reshape(input_shape)(input_tensor)
        input_transposed = tf.transpose(input_reshaped, perm=[0, 1, 3, 2])
        input_shuffled = Reshape(group_shape)(input_transposed)
        input_shuffled_transposed = tf.transpose(input_shuffled, perm=[0, 1, 3, 2])

        output_tensor = Reshape(keras.backend.int_shape(input_tensor))(input_shuffled_transposed)
        return output_tensor

    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_output = block_1(input_tensor=input_layer)
    main_path_output = block_2(input_tensor=main_path_output)
    main_path_output = block_3(input_tensor=main_path_output)

    output = keras.layers.add([main_path_output, branch_output])
    output = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model