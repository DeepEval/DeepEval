import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, DepthwiseConv2D, Flatten, Dense, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
        group2 = inputs_groups[1]
        output_tensor = Concatenate()([conv1_2, group2])
        return output_tensor
    block1_output = block1(conv1)

    def block2(input_tensor):
        shape = tf.shape(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 4, input_tensor.shape[-1] // 4))(input_tensor)
        permuted = Permute(axes=[0, 1, 3, 2])(reshaped)
        reshaped_back = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)
        return reshaped_back

    block2_output = block2(block1_output)
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model