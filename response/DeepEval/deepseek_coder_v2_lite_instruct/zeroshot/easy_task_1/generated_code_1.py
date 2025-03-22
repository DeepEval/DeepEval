import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the model
    model = Sequential()

    # Add the first convolutional layer followed by a max pooling layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the second convolutional layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # Add the first fully connected layer
    model.add(Dense(128, activation='relu'))

    # Add the second fully connected layer (output layer)
    model.add(Dense(10, activation='softmax'))

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()