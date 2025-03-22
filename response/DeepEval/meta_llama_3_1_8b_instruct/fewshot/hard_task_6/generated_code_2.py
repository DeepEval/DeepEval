import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate, BatchNormalization, Lambda, Reshape, DepthwiseConv2D
import tensorflow as tf

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
        shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, 64))(input_tensor)
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))(reshaped)
        reshaped_back = Reshape(target_shape=(shape[1], shape[2], 64))(permuted)
        return reshaped_back
    
    def block_3(input_tensor):
        return DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)
    block1_output_repeated = block_1(block3_output)

    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)

    main_path_output = Concatenate()([block1_output_repeated, block3_output, block2_output, block_path])

    flatten = Flatten()(main_path_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model