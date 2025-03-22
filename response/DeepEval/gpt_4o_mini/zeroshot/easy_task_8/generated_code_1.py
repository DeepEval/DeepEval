import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for MNIST images (28x28 pixels, 1 color channel)
    input_layer = layers.Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = layers.Dropout(0.5)(x)  # Dropout layer to mitigate overfitting

    # 1x1 convolutional layer
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout layer to mitigate overfitting

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout layer to mitigate overfitting

    # Output layer for 10 classes (digits 0-9)
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# You can now create the model and view its summary
model = dl_model()
model.summary()