import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Concatenate, Conv2D, MaxPooling2D, DepthwiseConv2D, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        reshaped = Reshape((shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
        transposed = tf.transpose(reshaped, perm=[0, 1, 3, 2])
        output_tensor = Reshape((shape[1], shape[2], shape[3]))(transposed)
        return output_tensor

    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    def branch(input_tensor):
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)
    branch_output = branch(input_tensor=input_layer)
    combined = keras.layers.Add()([branch_output, block3_output])

    output_layer = Dense(units=10, activation='softmax')(combined)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model