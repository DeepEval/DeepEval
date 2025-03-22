import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, GlobalMaxPooling2D, Reshape

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def transition_conv(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    def block_2(input_tensor):
        global_max_pooling = GlobalMaxPooling2D()(input_tensor)

        dense1 = Dense(units=32, activation='relu')(global_max_pooling)
        dense2 = Dense(units=32, activation='relu')(dense1)
        dense3 = Dense(units=32, activation='relu')(dense2)

        weights = Reshape(target_shape=(1, 1, 32))(dense3)

        main_path_output = tf.matmul(input_tensor, weights)

        branch_output = input_tensor

        output = tf.add(main_path_output, branch_output)

        output = Dense(units=10, activation='softmax')(output)

        return output

    block1_output = block_1(input_tensor=input_layer)
    transition_output = transition_conv(input_tensor=block1_output)
    block2_output = block_2(input_tensor=transition_output)

    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model