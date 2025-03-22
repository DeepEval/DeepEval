import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Concatenate()([main_path, main_path])
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Concatenate()([branch_path, branch_path])
    output_layer = Add()([main_path, branch_path])
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=128, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model