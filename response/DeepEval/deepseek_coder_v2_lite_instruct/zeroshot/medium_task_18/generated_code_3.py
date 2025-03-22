import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    model = models.Sequential()

    # 1st block: 1x1 convolution, 3x3 convolution, 5x5 convolution, max pooling
    model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))

    # 2nd block: 1x1 convolution, 3x3 convolution, 5x5 convolution, max pooling
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))

    # Flatten the features
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()