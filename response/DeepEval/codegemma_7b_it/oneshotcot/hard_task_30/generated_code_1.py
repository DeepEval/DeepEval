import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.initializers import GlorotUniform

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_main(input_tensor):
        path_main_conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=GlorotUniform(seed=1))(input_tensor)
        path_main_conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=GlorotUniform(seed=1))(path_main_conv_1)
        path_main_conv_3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=GlorotUniform(seed=1))(path_main_conv_2)
        path_main_add = Add()([path_main_conv_3, input_tensor])
        return path_main_add

    def block_branch(input_tensor):
        path_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=GlorotUniform(seed=1))(input_tensor)
        return path_branch

    block_main_output = block_main(input_tensor)
    block_branch_output = block_branch(input_tensor)
    block_main_branch_concat = Add()([block_main_output, block_branch_output])

    split_input = Lambda(tf.split, arguments={'num_or_size_splits': 3})(block_main_branch_concat)
    path_0 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=GlorotUniform(seed=1))(x))(split_input[0])
    path_1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=GlorotUniform(seed=1))(x))(split_input[1])
    path_2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer=GlorotUniform(seed=1))(x))(split_input[2])
    concat_output = Concatenate()([path_0, path_1, path_2])

    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model