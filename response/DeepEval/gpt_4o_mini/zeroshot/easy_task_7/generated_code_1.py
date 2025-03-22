import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    inputs = layers.Input(shape=input_shape)

    # Main path
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(10, (3, 3), padding='same', activation='relu')(x)  # Restore number of channels

    # Branch path (directly connected to input)
    branch = layers.Conv2D(10, (1, 1), padding='same', activation='relu')(inputs)

    # Combine main and branch paths
    combined = layers.add([x, branch])

    # Flatten and Fully connected layer
    x = layers.Flatten()(combined)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to compile the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()