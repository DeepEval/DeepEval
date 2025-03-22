import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        output_tensor = Concatenate()([pool1, pool2, pool3])
        return output_tensor

    block1_output = block(input_tensor=input_layer)
    dense1 = Dense(units=128, activation='relu')(block1_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()