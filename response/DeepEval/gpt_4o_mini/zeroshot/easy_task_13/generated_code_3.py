import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 images with 1 channel (grayscale)
    inputs = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer with dropout
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)

    # Second 1x1 convolutional layer with dropout
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)

    # 3x1 convolutional layer with dropout
    x = Conv2D(32, (3, 1), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)

    # 1x3 convolutional layer with dropout
    x = Conv2D(32, (1, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)

    # Restore the number of channels using another 1x1 convolutional layer
    x = Conv2D(1, (1, 1), activation='linear', padding='same')(x)

    # Add the processed features to the original input
    x = Add()([x, inputs])

    # Flattening the output
    x = Flatten()(x)

    # Fully connected layer for classification
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for digits 0-9

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model