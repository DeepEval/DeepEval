import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 channel

    # First 1x1 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    x = Dropout(0.2)(x)

    # Second 1x1 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)

    # 3x1 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(x)
    x = Dropout(0.2)(x)

    # 1x3 convolutional layer with dropout
    x = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(x)
    x = Dropout(0.2)(x)

    # Restore channels using a 1x1 convolutional layer
    x = Conv2D(filters=1, kernel_size=(1, 1), activation='linear')(x)

    # Add the original input to the processed features
    x = Add()([x, inputs])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer for classification
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout layer for regularization
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Print the model summary