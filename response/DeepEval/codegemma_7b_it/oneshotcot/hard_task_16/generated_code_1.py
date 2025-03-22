import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        path_1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        path_2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        path_3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)

        conv_1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path_1)
        conv_1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path_2)
        conv_1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path_3)

        concat_path = Concatenate()([conv_1_1, conv_1_2, conv_1_3])
        return concat_path

    # Transition Convolution
    def transition_conv(input_tensor):
        conv_trans = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv_trans

    # Block 2
    def block_2(input_tensor):
        conv_trans = transition_conv(input_tensor)
        batch_norm = BatchNormalization()(conv_trans)
        global_max_pool = MaxPooling2D(pool_size=(32, 32), strides=1, padding='valid')(batch_norm)

        dense_1 = Dense(units=128, activation='relu')(global_max_pool)
        dense_2 = Dense(units=32, activation='relu')(dense_1)

        reshape_weights = Reshape((32,))(dense_2)
        upsample_weights = UpSampling1D(size=32)(reshape_weights)
        upsample_weights = Reshape((32, 1))(upsample_weights)

        main_path_output = multiply([upsample_weights, batch_norm])
        branch_output = input_tensor

        final_output = Add()([main_path_output, branch_output])
        return final_output

    block_1_output = block_1(input_layer)
    block_2_output = block_2(block_1_output)

    flatten_layer = Flatten()(block_2_output)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model