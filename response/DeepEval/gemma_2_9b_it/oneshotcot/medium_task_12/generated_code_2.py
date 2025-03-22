import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  

    # Block 1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 2
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 3
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Concatenate blocks
    x = Concatenate()([x, ])  
    
    x = Flatten()(x)
    dense1 = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(dense1)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model