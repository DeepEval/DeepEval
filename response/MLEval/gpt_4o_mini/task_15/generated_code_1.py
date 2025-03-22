import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def method(batch_size=32, epochs=10):
    # Generate synthetic data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))  # Reshape for CNN
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    # Define a simple CNN model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # Return the final output (accuracy on the test set)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return test_loss, test_accuracy

# Example of calling the method with different batch sizes and epochs
output = method(batch_size=64, epochs=5)
print(f"Test Loss: {output[0]}, Test Accuracy: {output[1]}")