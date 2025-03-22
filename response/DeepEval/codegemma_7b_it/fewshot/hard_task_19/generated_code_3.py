import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    gap = GlobalAveragePooling2D()(main_path)
    fc1 = Dense(units=64, activation='relu')(gap)
    fc2 = Dense(units=32, activation='relu')(fc1)
    reshape_weights = Reshape(target_shape=(32, 1))(fc2)
    multiply_weights = keras.backend.expand_dims(reshape_weights, axis=1)
    multiply_output = keras.backend.multiply([main_path, multiply_weights])

    flatten_layer = Flatten()(multiply_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model