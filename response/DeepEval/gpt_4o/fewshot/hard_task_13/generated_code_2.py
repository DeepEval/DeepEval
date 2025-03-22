import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Feature extraction with parallel branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    block1_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Block 2: Dimensionality reduction and attention mechanism
    gap = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=block1_output.shape[-1], activation='relu')(gap)
    dense2 = Dense(units=block1_output.shape[-1], activation='sigmoid')(dense1)
    weights = Reshape((1, 1, block1_output.shape[-1]))(dense2)
    scaled_features = Multiply()([block1_output, weights])

    # Final Classification Layer
    flatten = GlobalAveragePooling2D()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model