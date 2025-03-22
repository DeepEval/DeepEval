import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Reshape, Conv2D, Permute, DepthwiseConv2D, Add, Concatenate, BatchNormalization

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate(axis=-1)([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        reshape = tf.reshape(input_tensor, (shape[0], shape[1], shape[2], 3, shape[3]//3))
        permute = Permute((0, 1, 3, 2, 4))(reshape)
        output_tensor = tf.reshape(permute, (shape[0], shape[1], shape[2], -1))
        return output_tensor

    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    branch_output = block_3(input_tensor)

    main_output = block_1(input_tensor)
    main_output = block_2(main_output)
    main_output = block_3(main_output)

    combined_output = Add()([main_output, branch_output])
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model