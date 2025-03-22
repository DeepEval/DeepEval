import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def method():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
    x_test = x_test.astype('float32') / 255.0    # Normalize to [0, 1]
    
    # Reshape the data to fit the model input
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))  # Add channel dimension
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))      # Add channel dimension

    # Convert labels to categorical one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Build the neural network model
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    output = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
    
    return output

# Call the method for validation
output = method()
print(output)