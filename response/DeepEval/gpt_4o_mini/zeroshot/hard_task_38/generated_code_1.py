import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def repeated_block(x):
    # Batch normalization
    x = layers.BatchNormalization()(x)
    # ReLU activation
    x = layers.ReLU()(x)
    # 3x3 Convolutional layer
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    return x

def pathway(input_tensor):
    x = input_tensor
    for _ in range(3):  # Repeat the block 3 times
        new_features = repeated_block(x)
        x = layers.concatenate([x, new_features])  # Concatenate along the channel dimension
    return x

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 with 1 channel
    inputs = layers.Input(shape=input_shape)

    # Two pathways
    pathway1_output = pathway(inputs)
    pathway2_output = pathway(inputs)

    # Concatenate outputs from both pathways
    merged = layers.concatenate([pathway1_output, pathway2_output])

    # Fully connected layers for classification
    x = layers.Flatten()(merged)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# To compile the model (optional)
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# To train the model, you would load the MNIST dataset and preprocess it accordingly:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1) / 255.0
x_test = np.expand_dims(x_test, -1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))