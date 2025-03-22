import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First parallel branch
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)

    # Second parallel branch
    branch2 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Parallel branch from input
    parallel_branch = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Addition operation
    added = Add()([branch1, branch2, parallel_branch])

    # Concatenation
    concatenated = Concatenate()([branch1, branch2, parallel_branch, added])

    # Flatten layer
    flattened = Flatten()(concatenated)

    # Fully connected layer
    dense = Dense(128, activation='relu')(flattened)

    # Output layer
    output_layer = Dense(10, activation='softmax')(dense)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()