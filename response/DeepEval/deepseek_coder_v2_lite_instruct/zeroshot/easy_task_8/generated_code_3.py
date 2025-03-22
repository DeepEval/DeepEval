import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():
    model = Sequential()

    # Depthwise Separable Convolution Layer
    model.add(DepthwiseConv2D(kernel_size=(3, 3), input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=1, kernel_size=(1, 1)))
    model.add(Dropout(0.25))

    # Flatten the output
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()