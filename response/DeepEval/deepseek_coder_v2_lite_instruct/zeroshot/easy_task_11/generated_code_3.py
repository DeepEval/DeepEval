import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    model = Sequential()

    # Add the first convolutional layer with average pooling
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(AveragePooling2D(pool_size=(5, 5)))

    # Add the second convolutional layer (1x1 convolution)
    model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))

    # Flatten the feature maps
    model.add(Flatten())

    # Add two fully connected layers with dropout for regularization
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer with softmax activation for multi-class classification
    model.add(Dense(10, activation='softmax'))

    return model

# Example usage:
# model = dl_model()
# model.summary()