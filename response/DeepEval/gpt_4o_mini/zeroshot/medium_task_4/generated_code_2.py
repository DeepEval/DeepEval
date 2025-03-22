import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Pathway 1: Two blocks of convolution followed by average pooling
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.AveragePooling2D(pool_size=(2, 2))(x1)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.AveragePooling2D(pool_size=(2, 2))(x1)

    # Pathway 2: Single convolutional layer
    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

    # Merge the two pathways
    merged = layers.add([x1, x2])

    # Flatten the output
    flat = layers.Flatten()(merged)

    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flat)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Optional: To compile the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()