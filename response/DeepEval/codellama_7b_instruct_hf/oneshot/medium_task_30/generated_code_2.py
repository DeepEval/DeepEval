import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First average pooling layer with pooling window and stride of 1x1
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    # Second average pooling layer with pooling window and stride of 2x2
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(pool1)
    # Third average pooling layer with pooling window and stride of 4x4
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(pool2)

    # Flatten the outputs of the three pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concatenated = keras.layers.concatenate([flatten1, flatten2, flatten3])

    # Flatten the concatenated output
    flat = Flatten()(concatenated)

    # Add two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model