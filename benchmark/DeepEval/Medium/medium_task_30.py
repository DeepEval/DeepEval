from keras.models import Model
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
        
    maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(maxpool1)

    maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(maxpool2)

    maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(maxpool3)

    concated = Concatenate()([flatten1, flatten2, flatten3])
    flatten_output = Flatten()(concated)
    dense = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
