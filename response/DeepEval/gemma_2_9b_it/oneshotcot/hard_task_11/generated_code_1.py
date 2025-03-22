import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel branch
    x_parallel = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_parallel = Concatenate()([
        Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(x_parallel),
        Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(x_parallel)
    ])
    
    x = Concatenate()([x, x_parallel])
    x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Direct connection
    x = Add()([input_layer, x])

    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model