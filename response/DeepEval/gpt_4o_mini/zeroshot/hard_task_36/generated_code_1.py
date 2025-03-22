import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for 28x28 grayscale images (MNIST)
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    # Convolutional layer with 3x3 kernel
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Two 1x1 convolutional layers
    x = Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same')(x)

    # Max pooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Dropout layer
    x = Dropout(0.5)(x)

    # Branch pathway
    # Convolutional layer to match the size of the feature maps from the main pathway
    branch = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

    # Fusion of main and branch pathways
    fused = Add()([x, branch])

    # Global average pooling layer
    x = GlobalAveragePooling2D()(fused)

    # Flattening layer
    x = Flatten()(x)

    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()