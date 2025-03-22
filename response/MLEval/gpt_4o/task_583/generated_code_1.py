import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def method():
    # Load MNIST data for demonstration
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
    # x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255

    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)

    # Create a Sequential model
    model = Sequential()

    # Input layer with 512 neurons and ReLU activation
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    # Add dropout layer
    model.add(Dropout(0.2))

    # Hidden layer with 256 neurons and ReLU activation
    model.add(Dense(256, activation='relu'))
    # Add dropout layer
    model.add(Dropout(0.2))

    # Output layer with 10 neurons (one for each class) and softmax activation
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    # model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=2)

    # # Evaluate the model
    # test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # # Prepare the output
    # output = {
    #     'model': model,
    #     'test_loss': test_loss,
    #     'test_accuracy': test_accuracy
    # }
    
    # return output

# Call the method for validation
# result = method()
# print(f"Test Loss: {result['test_loss']:.4f}")
# print(f"Test Accuracy: {result['test_accuracy']:.4f}")
method()