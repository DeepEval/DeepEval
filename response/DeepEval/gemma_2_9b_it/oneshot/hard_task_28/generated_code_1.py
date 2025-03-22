import keras
from keras.layers import Input, Conv2D, LayerNormalization, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    depthwise = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='valid', use_bias=False, activation='relu')(input_layer)
    norm = LayerNormalization()(depthwise)
    pointwise1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(norm)
    pointwise2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(pointwise1)

    # Branch Path
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer) 

    # Combine Paths
    combined = Concatenate()([pointwise2, branch])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model