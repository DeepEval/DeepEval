import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)
    branch_path = Input(shape=(28, 28, 1))
    adding_layer = Add()([main_path, branch_path])

    # Second block
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(adding_layer)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(maxpool1)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(maxpool2)
    flatten = Flatten()(maxpool3)

    # Final classification
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model