import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dense, Reshape
import tensorflow as tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)

        conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3_2)

        output_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])
        return output_tensor

    block1_output = block_1(input_layer)
    transition_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    def block_2(input_tensor):
        global_max_pool = MaxPooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(input_tensor)
        dense1 = Dense(units=128, activation='relu')(global_max_pool)
        dense2 = Dense(units=128, activation='relu')(dense1)
        weights = Reshape(target_shape=(128, 1, 1))(dense2)
        output_tensor = tf.multiply(input_tensor, weights)
        return output_tensor

    main_path_output = block_2(transition_conv)
    branch_output = Input(shape=(32, 32, 3))
    
    merged_output = Add()([main_path_output, branch_output])
    output_layer = Dense(units=10, activation='softmax')(merged_output)
    model = keras.Model(inputs=[input_layer, branch_output], outputs=output_layer)

    return model