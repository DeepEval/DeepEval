import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 color channel

    def conv_block(x):
        # Batch Normalization
        x = layers.BatchNormalization()(x)
        # ReLU Activation
        x = layers.ReLU()(x)
        # 3x3 Convolution
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        return x

    # Pathway 1
    pathway1 = inputs
    for _ in range(3):  # Repeat the block structure 3 times
        conv_output = conv_block(pathway1)
        pathway1 = layers.Concatenate()([pathway1, conv_output])  # Concatenate along channels

    # Pathway 2 (same structure as pathway 1)
    pathway2 = inputs
    for _ in range(3):  # Repeat the block structure 3 times
        conv_output = conv_block(pathway2)
        pathway2 = layers.Concatenate()([pathway2, conv_output])  # Concatenate along channels

    # Merge both pathways
    merged = layers.Concatenate()([pathway1, pathway2])

    # Global Average Pooling
    pooled = layers.GlobalAveragePooling2D()(merged)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(pooled)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST digits

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))