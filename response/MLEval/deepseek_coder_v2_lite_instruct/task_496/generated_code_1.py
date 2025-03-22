import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def method():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the input data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build the neural network model
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),  # Flatten the input image
        layers.Dense(128, activation='relu'),  # First hidden layer with 128 neurons and ReLU activation
        layers.Dropout(0.2),                   # Dropout layer to prevent overfitting
        layers.Dense(10, activation='softmax') # Output layer with 10 neurons (one for each digit) and softmax activation
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Return the model's predictions for the test set
    predictions = model.predict(x_test)
    return predictions

# Call the method for validation
output = method()
print(output)