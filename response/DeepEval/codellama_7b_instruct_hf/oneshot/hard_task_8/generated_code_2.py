import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, Concatenate, Permute, Reshape, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
    x = DepthwiseSeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    x = Concatenate()([x, x])

    # Block 2
    x = Reshape((28, 28, 2, 16))(x)
    x = Permute((1, 2, 3, 4))(x)
    x = Reshape((28, 28, 16))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model