import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    x1 = input_layer
    for _ in range(3):
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)
        x1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x1 = Concatenate()([x1, input_layer]) 

    # Pathway 2
    x2 = input_layer
    for _ in range(3):
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)
        x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
        x2 = Concatenate()([x2, input_layer])

    # Merge pathways
    merged = Concatenate()([x1, x2])
    
    # Classification layers
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model