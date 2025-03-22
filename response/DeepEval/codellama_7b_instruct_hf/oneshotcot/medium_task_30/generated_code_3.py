import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Three average pooling layers with pooling windows and strides of 1x1, 2x2, and 4x4
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(pool1)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(pool2)

    # Flatten the outputs of the pooling layers
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concat = Concatenate()([flat1, flat2, flat3])

    # Flatten the concatenated output
    flat_concat = Flatten()(concat)

    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat_concat)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model