import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Second convolutional layer
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    
    # Max pooling layer
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Skip connection: add the input to the output of the max pooling layer
    x = layers.add([x, inputs])

    # Flatten the feature map
    x = layers.Flatten()(x)

    # First fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    
    # Second fully connected layer
    x = layers.Dense(64, activation='relu')(x)

    # Output layer with softmax activation for multi-class classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()