import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def channel_split(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        return tf.split(input_tensor, shape[3] // 3, axis=3)

    groups = Lambda(channel_split)(input_layer)
    conv_group_1 = Conv2D(filters=shape[3] // 9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
    conv_group_2 = Conv2D(filters=shape[3] // 9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[1])
    conv_group_3 = Conv2D(filters=shape[3] // 9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[2])

    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group_1)
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group_2)
    max_pooling_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group_3)

    concat = Concatenate()([max_pooling, max_pooling_2, max_pooling_3])
    bath_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model