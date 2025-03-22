import keras
from keras.layers import Input, Dense, Flatten, Conv2D, GlobalAveragePooling2D, Add


def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Dense(units=128, activation='relu')(main_path)
    main_path = Dense(units=10, activation='softmax')(main_path)
    branch_path = Dense(units=128, activation='relu')(input_layer)
    branch_path = Dense(units=10, activation='softmax')(branch_path)
    model = Add()([main_path, branch_path])
    output_layer = Dense(units=10, activation='softmax')(model)
    return keras.Model(inputs=input_layer, outputs=output_layer)