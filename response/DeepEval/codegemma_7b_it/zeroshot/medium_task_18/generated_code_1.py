import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Extract features at different scales
    x1 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)
    x1 = layers.MaxPooling2D(pool_size=2)(x1)

    x2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    x2 = layers.MaxPooling2D(pool_size=2)(x2)

    x3 = layers.Conv2D(filters=32, kernel_size=5, activation='relu')(inputs)
    x3 = layers.MaxPooling2D(pool_size=2)(x3)

    # Concatenate features
    concat = layers.concatenate([x1, x2, x3])

    # Fully connected layers
    flatten = layers.Flatten()(concat)
    outputs = layers.Dense(units=10, activation='softmax')(flatten)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
print(model.summary())