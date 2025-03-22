# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Block 1: Three parallel paths with different pooling scales
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    path1 = layers.GlobalAveragePooling2D()(x)
    path2 = layers.GlobalAveragePooling2D(strides=(2, 2))(x)
    path3 = layers.GlobalAveragePooling2D(strides=(4, 4))(x)

    # Concatenate the outputs from the three paths
    x = layers.Concatenate()([path1, path2, path3])
    x = layers.Flatten()(x)

    # Apply a fully connected layer
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Reshape((16, 4))(x)

    # Block 2: Three feature extraction branches
    branch1 = layers.Conv2D(16, (1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(16, (3, 3), activation='relu')(x)
    branch3 = layers.Conv2D(16, (1, 7), activation='relu')(x)
    branch3 = layers.Conv2D(16, (7, 1), activation='relu')(branch3)
    branch3 = layers.Conv2D(16, (3, 3), activation='relu')(branch3)

    # Apply average pooling in the third branch
    branch3 = layers.AveragePooling2D((3, 3))(branch3)

    # Concatenate the outputs from all branches
    x = layers.Concatenate()([branch1, branch2, branch3])
    x = layers.Flatten()(x)

    # Apply two fully connected layers for classification
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Build the model
model = dl_model()
model.summary()