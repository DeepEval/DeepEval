# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Define the neural network model
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Test accuracy: {accuracy:.2f}')

# Define the method function
def method():
    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f'Test accuracy: {accuracy:.2f}')
    
    return accuracy

# Call the method function for validation
output = method()