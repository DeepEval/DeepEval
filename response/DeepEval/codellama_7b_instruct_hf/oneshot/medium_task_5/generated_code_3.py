import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    combined_output = Concatenate()([main_path, branch_path])

    bath_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model