import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28,28,1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        output_tensor = Concatenate()([conv3, conv4])
        return output_tensor

    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        input_tensor = tf.reshape(input_tensor, (shape[0], shape[1], shape[2], 2, shape[3]//2))
        input_tensor = tf.transpose(input_tensor, perm=[0,1,3,2,4])
        input_tensor = tf.reshape(input_tensor, (shape[0], shape[1], 2, shape[2]*shape[3]//2))
        input_tensor = tf.transpose(input_tensor, perm=[0,1,3,2])
        return input_tensor

    block1_output = block_1(input_tensor=conv)
    block2_output = block_2(input_tensor=block1_output)
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model