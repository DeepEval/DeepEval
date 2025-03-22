import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    channel_split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    path1 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(channel_split[0])
    path2 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(channel_split[1])
    path3 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x))(channel_split[2])
    path4 = Lambda(lambda x: MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x))(channel_split[0])

    concat_output = Concatenate()([path1, path2, path3, path4])
    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model