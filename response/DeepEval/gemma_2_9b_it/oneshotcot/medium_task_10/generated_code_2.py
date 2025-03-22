import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        shortcut = input_tensor
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        output_tensor = Add()([x, shortcut])
        return output_tensor

    # Level 1
    x = basic_block(x)

    # Level 2
    x = basic_block(x)
    x = basic_block(x)

    # Level 3
    branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = basic_block(x)
    x = Add()([branch, x]) 

    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model