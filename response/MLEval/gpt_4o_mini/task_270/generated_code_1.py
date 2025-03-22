import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def method():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)  # One-hot encoding
    y_test = to_categorical(y_test, num_classes=10)

    # Create the MLP model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # Flatten the input
    model.add(Dense(128, activation='relu'))  # Hidden layer with 128 neurons
    model.add(Dense(10, activation='softmax'))  # Output layer with 10 classes

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)  # Adjust epochs and batch size as needed

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    
    output = {'loss': loss, 'accuracy': accuracy}
    return output

# Call the method for validation
output = method()
print(output)