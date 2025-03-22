import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# ENCODING_DIM will be a value you choose, e.g., 64 or 128
ENCODING_DIM = 64

def method():
    # Load the MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))

    # Define the encoder model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(ENCODING_DIM, activation='relu')
    ])

    return model(x_train[:1]).numpy()

# Call the method for validation
output = method()
print("Output of the encoder:", output)