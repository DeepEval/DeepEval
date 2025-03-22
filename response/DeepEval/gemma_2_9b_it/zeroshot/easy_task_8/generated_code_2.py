from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),  # Input layer for MNIST images (28x28 pixels, 1 channel)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # Depthwise separable convolutional layer
            layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same'),  # 1x1 convolutional layer
            layers.Conv2D(32, (1, 1), activation='relu', padding='same'),  # 1x1 convolutional layer
            layers.Dropout(0.25),  # Dropout layer
            layers.MaxPooling2D((2, 2), padding='same'),  # Max pooling layer
            layers.Flatten(),  # Flatten the output for the fully connected layer
            layers.Dense(10, activation='softmax')  # Output layer with 10 units (for MNIST classes)
        ]
    )
    return model