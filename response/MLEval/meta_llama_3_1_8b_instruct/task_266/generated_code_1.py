# Import necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate random data for demonstration
np.random.seed(0)
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
def method(batch_size, epochs):
    # Reshape data
    X_train = X_train.reshape(-1, 10, 1)
    X_test = X_test.reshape(-1, 10, 1)

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(10, 1)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_split=0.2, 
                        verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy: {accuracy:.2f}')

    return history, model

# Call the generated method for validation
batch_size = 32
epochs = 10
history, model = method(batch_size, epochs)

# Optional: Print the output
# output = history.history['accuracy'][-1]
# print(output)