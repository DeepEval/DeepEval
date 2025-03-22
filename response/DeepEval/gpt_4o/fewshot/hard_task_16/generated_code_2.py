import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, GlobalMaxPooling2D, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        conv11_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv11_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv11_1)
        conv11_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv11_2)

        conv21_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv21_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv21_1)
        conv21_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv21_2)

        conv31_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv31_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv31_1)
        conv31_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv31_2)

        output_tensor = Concatenate()([conv11_3, conv21_3, conv31_3])
        return output_tensor

    block1_output = block_1(input_layer)

    transition_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    def block_2(input_tensor):
        global_pool = GlobalMaxPooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(global_pool)
        dense2 = Dense(units=3, activation='sigmoid')(dense1)

        reshaped_weights = Lambda(lambda x: tf.reshape(x, (1, 1, 1, -1)))(dense2)

        weighted_tensor = Multiply()([input_tensor, reshaped_weights])

        return weighted_tensor

    block2_output = block_2(transition_conv)

    # Direct branch from input
    branch_output = input_layer

    # Add main path and branch outputs
    adding_layer = Add()([block2_output, branch_output])

    # Final classification layer
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model