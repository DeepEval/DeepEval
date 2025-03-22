import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for MNIST images (28x28 grayscale images)
    input_layer = layers.Input(shape=(28, 28, 1))

    # 1x1 Convolutional Layer to increase dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 Depthwise Separable Convolutional Layer for feature extraction
    x = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # 1x1 Convolutional Layer to reduce dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)

    # Adding the output from the last layer to the input layer (skip connection)
    x = layers.Add()([x, input_layer])

    # Flattening the output for the fully connected layer
    x = layers.Flatten()(x)

    # Fully connected layer to generate the final classification probabilities
    output_layer = layers.Dense(units=10, activation='softmax')(x)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()