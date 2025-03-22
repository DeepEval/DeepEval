import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape based on MNIST data (28x28 pixels, 1 channel)
    input_shape = (28, 28, 1)

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # First convolutional layer followed by max pooling
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Second convolutional layer
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)

    # Flatten the feature maps to 1D vector
    x = layers.Flatten()(x)

    # First fully connected layer
    x = layers.Dense(128, activation='relu')(x)

    # Second fully connected layer
    x = layers.Dense(64, activation='relu')(x)

    # Output layer for classification (10 classes for digits 0-9)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

# Example usage: create the model
model = dl_model()
model.summary()

# Load and preprocess MNIST data (for testing the model)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Fit the model (optional, uncomment if you want to train the model)
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)