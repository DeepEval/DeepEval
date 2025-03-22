import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x1)

    x1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
    x1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x1)

    # Path 2
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate outputs
    x = Concatenate()([x1, x2])

    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model