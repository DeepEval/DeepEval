import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First branch
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)

    # Second branch
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)

    # Parallel branch
    parallel_branch = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Addition operation
    added_output = Add()([branch1, branch2, parallel_branch])

    # Concatenate the outputs from the two blocks
    concatenated_output = Concatenate()([branch1, branch2, parallel_branch])

    # Flatten the concatenated output
    flattened_output = Flatten()(concatenated_output)

    # Fully connected layer
    fc_output = Dense(128, activation='relu')(flattened_output)

    # Output layer
    output_layer = Dense(10, activation='softmax')(fc_output)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model