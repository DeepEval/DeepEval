import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, GlobalAveragePooling2D, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: Global average pooling
    avg_pool = GlobalAveragePooling2D()(input_layer)

    # Dense layer 1
    dense1 = Dense(units=1024, activation='relu')(avg_pool)

    # Dense layer 2
    dense2 = Dense(units=512, activation='relu')(dense1)

    # Reshape weights to match input shape
    reshaped_weights = Reshape((1024, 2, 2))(dense2)

    # Element-wise multiplication with the block's input
    multiplied_input = keras.backend.batch_dot(input_layer, reshaped_weights, axes=(2, 2))

    # Branch 2: Continue processing without global average pooling
    dense3 = Dense(units=1024, activation='relu')(input_layer)

    dense4 = Dense(units=512, activation='relu')(dense3)

    # Concatenate outputs from both branches
    concatenated_output = Concatenate()([multiplied_input, dense4])

    # Flatten layer
    flattened = Flatten()(concatenated_output)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model