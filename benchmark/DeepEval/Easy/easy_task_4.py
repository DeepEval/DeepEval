from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_layer)
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv)
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv)
    pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool)
    conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv)
    conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv)
    conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv)
    pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    flatten = Flatten()(pool)
    dense = Dense(units=2048, activation='relu')(flatten)
    dense = Dense(units=1024, activation='relu')(dense)
    output_layer = Dense(units=100, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

