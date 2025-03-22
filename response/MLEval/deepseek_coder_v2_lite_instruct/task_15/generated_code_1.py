import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
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
    
    # Experiment with different batch sizes and number of epochs
    batch_sizes = [32, 64, 128]
    epochs = [5, 10, 20]
    
    for bs in batch_sizes:
        for e in epochs:
            print(f"Training with batch size {bs} and {e} epochs...")
            model.fit(x_train, y_train, batch_size=bs, epochs=e, validation_data=(x_test, y_test))
    
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    
    # Return the final output (accuracy in this case)
    return accuracy

# Call the method for validation
final_output = method()
print(f"Final accuracy: {final_output}")