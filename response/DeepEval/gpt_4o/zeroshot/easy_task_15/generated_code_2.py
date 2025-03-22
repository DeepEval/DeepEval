import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with a single channel

    # Input layer
    inputs = Input(shape=input_shape)

    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Second block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Flattening layer
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)  # MNIST has 10 classes

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()