import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    conv_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_main)

    concat_layer = Add()([conv1_2, avg_pool])

    def block_2(input_tensor):

        global_avg_pool = keras.layers.GlobalAveragePooling2D()(input_tensor)
        global_avg_pool = Dense(units=64, activation='relu')(global_avg_pool)
        global_avg_pool = Dense(units=64, activation='relu')(global_avg_pool)
        output_tensor = Reshape(target_shape=(1, 1, 64))(global_avg_pool)
        output_tensor = keras.layers.Lambda(lambda x: x * input_tensor)(output_tensor)
        output_tensor = Flatten()(output_tensor)
        output_tensor = Dense(units=10, activation='softmax')(output_tensor)
        return output_tensor

    block2_output = block_2(avg_pool)
    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model