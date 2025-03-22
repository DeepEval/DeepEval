import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def dl_model():
    # Input layer for 28x28 images with 1 channel (grayscale)
    inputs = layers.Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer to reduce dimensionality
    x = layers.Conv2D(16, (1, 1), activation='relu')(inputs)

    # 3x3 convolutional layer to extract features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Second 1x1 convolutional layer to restore dimensionality
    x = layers.Conv2D(16, (1, 1), activation='relu')(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer with 10 neurons for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create and summarize the model
model = dl_model()
model.summary()

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')