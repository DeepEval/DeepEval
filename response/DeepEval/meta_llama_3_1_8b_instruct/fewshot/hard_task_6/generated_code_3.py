import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Permute

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Dense(32, activation='relu')(inputs_groups[0])
        conv2 = Dense(32, activation='relu')(inputs_groups[1])
        conv3 = Dense(32, activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, shape[-1]//3))(input_tensor)
        permuted = Permute((3, 4, 1, 2))(reshaped)
        output_tensor = Reshape(target_shape=(shape[1], shape[2], shape[-1]//3))(permuted)
        return output_tensor

    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    main_path = block_1(input_layer)
    main_path = block_2(main_path)
    main_path = block_3(main_path)
    main_path = block_1(main_path)

    branch_path = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)

    combined_output = Concatenate()([main_path, branch_path])

    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model