import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape for MNIST (28x28 grayscale images)
    input_shape = (28, 28, 1)

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # First convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # Max pooling layer
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Second convolutional layer
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)

    # Flatten the feature maps to a 1D vector
    x = layers.Flatten()(x)

    # First fully connected layer
    x = layers.Dense(128, activation='relu')(x)

    # Second fully connected layer
    x = layers.Dense(64, activation='relu')(x)

    # Output layer with softmax activation for classification (10 classes)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model using the Functional API
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to load the MNIST data and train the model
def load_and_train_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0  # Reshape and normalize
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Create the model
    model = dl_model()

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model

# Uncomment the following line to train the model
# trained_model = load_and_train_model()