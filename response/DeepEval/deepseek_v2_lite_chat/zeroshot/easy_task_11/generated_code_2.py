import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Define model architecture
    model = Sequential([
        # Convolutional layer
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        # Average Pooling layer
        MaxPooling2D(pool_size=(3, 3)),
        # Flatten layer
        Flatten(),
        # First fully connected layer
        Dense(128, activation='relu'),
        # Dropout layer to prevent overfitting
        Dropout(0.5),
        # Second fully connected layer
        Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Build the model
model = dl_model()