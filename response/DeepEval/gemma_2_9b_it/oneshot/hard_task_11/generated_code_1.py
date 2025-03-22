import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel branch
    x2_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x2_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x2 = Concatenate()([x2_1, x2_2])

    # Concatenate outputs and reduce dimensions
    x = Concatenate()([x1, x2])
    x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Direct connection
    x = Add()([input_layer, x])

    # Flatten and classification layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model