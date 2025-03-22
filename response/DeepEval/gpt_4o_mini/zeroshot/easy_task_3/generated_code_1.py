import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel (grayscale)
    inputs = Input(shape=input_shape)

    # First Convolutional Block
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second Convolutional Block
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flattening the feature maps
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # 10 classes for digits 0-9

    # Constructing the model
    model = Model(inputs=inputs, outputs=x)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # To display the model architecture