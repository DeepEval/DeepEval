import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First average pooling layer with a 1x1 window
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Second average pooling layer with a 2x2 window
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Third average pooling layer with a 4x4 window
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten the outputs of each pooling layer
    flatten_1x1 = Flatten()(avg_pool_1x1)
    flatten_2x2 = Flatten()(avg_pool_2x2)
    flatten_4x4 = Flatten()(avg_pool_4x4)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])

    # Further flatten the concatenated output (though it should already be flattened)
    # and pass it through two fully connected layers
    dense1 = Dense(units=256, activation='relu')(concatenated)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Output layer with 10 units for the CIFAR-10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model