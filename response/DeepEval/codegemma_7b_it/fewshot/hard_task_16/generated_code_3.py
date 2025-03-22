import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def transition_conv(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        return conv

    def block_2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        kernel = tf.Variable(tf.random.normal([shape[3], shape[3]]))
        biases = tf.Variable(tf.random.normal([shape[3]]))
        branch_out = tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME') + biases
        branch_out = tf.reshape(branch_out, (shape[0], shape[1], shape[2], shape[3]))
        return branch_out

    block1_output = block_1(input_tensor=input_layer)
    transition_output = transition_conv(input_tensor=block1_output)
    block2_output = block_2(input_tensor=transition_output)

    main_path = block2_output
    branch_path = input_layer

    adding_layer = keras.layers.Add()([main_path, branch_path])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model