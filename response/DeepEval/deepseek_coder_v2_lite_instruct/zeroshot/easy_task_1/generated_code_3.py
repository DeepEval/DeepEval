import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # First fully connected layer
    model.add(Dense(128, activation='relu'))

    # Second fully connected layer (output layer)
    model.add(Dense(10, activation='softmax'))

    return model