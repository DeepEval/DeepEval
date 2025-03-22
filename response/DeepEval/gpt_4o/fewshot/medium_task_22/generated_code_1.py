import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 Convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
    branch2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_1x1)
    branch2_3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_3x3_1)

    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2_3x3_2, branch3])

    # Flatten and Dense Layers for Classification
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model