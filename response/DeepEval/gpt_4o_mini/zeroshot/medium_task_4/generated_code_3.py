import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 32x32x3 images
    input_layer = layers.Input(shape=(32, 32, 3))

    # Pathway 1
    # Block 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.AveragePooling2D(pool_size=(2, 2))(x1)

    # Block 2
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.AveragePooling2D(pool_size=(2, 2))(x1)

    # Pathway 2
    x2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)

    # Combine both pathways
    combined = layers.add([x1, x2])

    # Flatten the combined output
    x = layers.Flatten()(combined)

    # Fully connected layer to output probabilities for 10 classes
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
model = dl_model()
model.summary()