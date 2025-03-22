import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    main_path = input_layer
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    for kernel_size in [1, 3, 5]:
        main_path = Conv2D(filters=32, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(main_path)
        branch_path = Conv2D(filters=32, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(branch_path)

    main_path = Concatenate()([main_path, branch_path])
    main_path = BatchNormalization()(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(units=128, activation='relu')(main_path)
    main_path = Dense(units=64, activation='relu')(main_path)
    output_layer = Dense(units=10, activation='softmax')(main_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model