import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main Path
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch Path
    # Block 1
    y = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    y = layers.MaxPooling2D((2, 2))(y)

    # Combining paths
    combined = layers.add([x, y])

    # Flattening layer
    flattened = layers.Flatten()(combined)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    # Constructing the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()