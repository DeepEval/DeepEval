import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def method():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Define the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define the batch sizes and epochs to experiment with
    # batch_sizes = [32, 64, 128]
    # epochs = [10, 20, 30]

    # 修改
    batch_sizes = [32]
    epochs = [10]
    
    best_accuracy = 0
    best_params = {}
    
    # Experiment with different batch sizes and epochs
    for bs in batch_sizes:
        for e in epochs:
            print(f"Training with batch size {bs} and {e} epochs...")
            model.fit(x_train, y_train, batch_size=bs, epochs=e, validation_data=(x_test, y_test))
            
            # Evaluate the model
            loss, accuracy = model.evaluate(x_test, y_test)
            print(f"Batch size {bs}, epochs {e}: Test loss = {loss}, Test accuracy = {accuracy}")
            
            # Update the best parameters if the current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'batch_size': bs, 'epochs': e}
    
    # Return the best parameters and the final output
    output = {
        'best_accuracy': best_accuracy,
        'best_params': best_params
    }
    
    return output

# Call the method for validation
output = method()
print(output)