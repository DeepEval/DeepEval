import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flat1 = Flatten()(avgpool1)
        flat2 = Flatten()(avgpool2)
        flat3 = Flatten()(avgpool3)
        concat = Concatenate()([flat1, flat2, flat3])
        dense1 = Dense(units=128, activation='relu')(concat)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        model = keras.Model(inputs=input_tensor, outputs=output_layer)
        return model

    model = block(input_layer)

    return model