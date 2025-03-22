import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    x = input_layer
    for _ in range(3):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = Concatenate()([x, input_layer]) 

    # Pathway 2
    y = input_layer
    for _ in range(3):
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = Concatenate()([y, input_layer])

    # Merge pathways
    merged = Concatenate()([x, y])
    
    # Classification layers
    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model