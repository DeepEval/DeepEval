# Import necessary libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

# Create the deep learning model
def dl_model():
    # Load the MNIST dataset
    x_train, y_train, x_test, y_test = load_data()

    # Define the input shape
    input_shape = (28, 28, 1)

    # Create the deep learning model
    model = keras.Sequential([
        # Reduce the input dimensionality with a 1x1 convolution
        layers.Conv2D(32, (1, 1), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Extract features using a 3x3 convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Restore the dimensionality of the feature map with another 1x1 convolution layer
        layers.Conv2D(64, (1, 1), activation='relu'),

        # Flatten the output
        layers.Flatten(),

        # Add a fully connected layer with 10 neurons for classification
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model

# Usage
model = dl_model()
model.summary()