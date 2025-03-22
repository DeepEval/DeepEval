import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = layers.GlobalAveragePooling2D()(branch1)
    branch1 = layers.Dense(64, activation='relu')(branch1)
    branch1_weights = layers.Dense(32)(branch1)
    branch1_weights = layers.Reshape((1, 1, 32))(branch1_weights)
    branch1_output = layers.multiply([inputs, branch1_weights])

    # Branch 2
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = layers.GlobalAveragePooling2D()(branch2)
    branch2 = layers.Dense(64, activation='relu')(branch2)
    branch2_weights = layers.Dense(32)(branch2)
    branch2_weights = layers.Reshape((1, 1, 32))(branch2_weights)
    branch2_output = layers.multiply([inputs, branch2_weights])

    # Concatenate outputs from both branches
    concatenated = layers.Concatenate()([branch1_output, branch2_output])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # Fully connected layer
    output = layers.Dense(10, activation='softmax')(flattened)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# Example usage:
# model = dl_model()
# model.summary()