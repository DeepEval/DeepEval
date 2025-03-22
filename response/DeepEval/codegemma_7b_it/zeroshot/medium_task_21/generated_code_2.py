import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Define the four branches
    branch1 = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(128, (3, 3), activation='relu')(branch2)
    branch3 = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(128, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(128, (3, 3), activation='relu')(branch3)
    branch4 = layers.AveragePooling2D()(inputs)
    branch4 = layers.Conv2D(64, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs from all branches
    concat = layers.concatenate([branch1, branch2, branch3, branch4])

    # Apply dropout to prevent overfitting
    dropout = layers.Dropout(0.4)(concat)

    # Pass the concatenated output through three fully connected layers
    fc1 = layers.Dense(512, activation='relu')(dropout)
    fc2 = layers.Dense(256, activation='relu')(fc1)
    outputs = layers.Dense(10, activation='softmax')(fc2)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model