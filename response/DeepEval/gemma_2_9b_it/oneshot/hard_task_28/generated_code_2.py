import keras
from keras.layers import Input, Conv2D, BatchNormalization, LayerNormalization, Concatenate, Flatten, Dense

def dl_model():     
    input_tensor = Input(shape=(32, 32, 3))

    # Main Path
    depthwise = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')(input_tensor)
    norm = LayerNormalization()(depthwise)
    pointwise1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(norm)
    pointwise2 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(pointwise1)

    # Branch Path
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)

    # Combine Paths
    combined = Concatenate()([pointwise2, branch])

    # Flatten and Dense Layers
    flatten = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)

    return model