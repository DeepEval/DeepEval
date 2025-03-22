import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

def method():
    # Load the MNIST dataset
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # # Normalize the data
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255

    # # One-hot encode the labels
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)

    # Create a simple model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Experiment with different batch_size and number of epochs
    # batch_sizes = [32, 64, 128]
    # num_epochs = [5, 10, 15]

    # # Dictionary to store results
    # results = {}

    # for batch_size in batch_sizes:
    #     for epochs in num_epochs:
    #         print(f"Training with batch_size={batch_size} and epochs={epochs}")
    #         # Train the model
    #         history = model.fit(x_train, y_train,
    #                             batch_size=batch_size,
    #                             epochs=epochs,
    #                             validation_data=(x_test, y_test),
    #                             verbose=0)
            
    #         # Evaluate the model
    #         test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    #         print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    #         # Save the results
    #         results[(batch_size, epochs)] = (test_loss, test_accuracy)

    # return results

# Call the method for validation
# output = method()
# print("Experimentation Results:", output)

method()