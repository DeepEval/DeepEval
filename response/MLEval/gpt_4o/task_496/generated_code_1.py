import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def method():
    # Load the MNIST dataset
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # # Preprocess the data
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)

    # Build the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    # model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    # # Evaluate the model on the test data
    # test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # # Output the test accuracy
    # output = test_accuracy
    # return output

# Call the method for validation
# accuracy = method()
# print("Test accuracy:", accuracy)
method()