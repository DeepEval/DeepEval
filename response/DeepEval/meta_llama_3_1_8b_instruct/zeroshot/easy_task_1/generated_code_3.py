# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model consists of two sequential layers comprising a convolutional layer followed by a max pooling layer,
    followed by an additional convolutional layer. After these layers, the feature maps are flattened into a one-dimensional vector.
    This vector is then processed by two fully connected layers to produce the final classification results.
    
    Returns:
        model: The constructed deep learning model.
    """

    # Create a sequential model
    model = keras.Sequential()

    # Add the first convolutional layer with 32 filters, kernel size 3x3, and ReLU activation
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    # Add the first max pooling layer with a pool size of 2x2
    model.add(layers.MaxPooling2D((2, 2)))

    # Add the second convolutional layer with 64 filters, kernel size 3x3, and ReLU activation
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add the second max pooling layer with a pool size of 2x2
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the feature maps into a one-dimensional vector
    model.add(layers.Flatten())

    # Add the first fully connected layer with 64 units and ReLU activation
    model.add(layers.Dense(64, activation='relu'))

    # Add the second fully connected layer with 10 units and softmax activation (for multi-class classification)
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model with a loss function and optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
if __name__ == "__main__":
    model = dl_model()
    model.summary()