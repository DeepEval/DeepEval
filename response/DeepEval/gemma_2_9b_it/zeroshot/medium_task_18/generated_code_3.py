from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Layer 1: 3x3 Convolutional layer with 32 filters, ReLU activation
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((3, 3)))

    # Layer 2: 1x1 Convolutional layer with 64 filters, ReLU activation
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))

    # Layer 3: 5x5 Convolutional layer with 64 filters, ReLU activation
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))

    # Concatenate features from different convolutional layers
    model.add(layers.Concatenate()) 

    # Flatten the output for fully connected layers
    model.add(layers.Flatten())

    # Layer 4: Fully connected layer with 128 units, ReLU activation
    model.add(layers.Dense(128, activation='relu'))

    # Layer 5: Output layer with 10 units (for CIFAR-10 classes) and softmax activation
    model.add(layers.Dense(10, activation='softmax'))

    return model