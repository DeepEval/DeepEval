import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

def method():
    # Define the encoding dimension
    ENCODING_DIM = 32

    # Load the MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))

    # Define the encoder model
    encoder = Sequential()
    encoder.add(Dense(512, activation='relu', input_shape=(784,)))
    encoder.add(Dense(256, activation='relu'))
    encoder.add(Dense(ENCODING_DIM, activation='relu'))

    # Get the output of the encoder
    output = encoder(x_test[:1])  # Use the first test example to get the output

    return output

# Call the method for validation
output = method()
print(output)