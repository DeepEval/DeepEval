import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    main_path = input_layer
    for _ in range(2):
        main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Dropout(rate=0.25)(main_path)

    main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    branch_path = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    output_path = keras.layers.add([main_path, branch_path])

    output_path = BatchNormalization()(output_path)
    output_path = Flatten()(output_path)
    output_path = Dense(units=128, activation='relu')(output_path)

    output_layer = Dense(units=10, activation='softmax')(output_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model