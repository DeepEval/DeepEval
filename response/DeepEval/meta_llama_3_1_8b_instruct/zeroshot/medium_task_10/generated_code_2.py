# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dl_model():
    # Define the input shape and create an input layer
    input_shape = (32, 32, 3)
    inputs = keras.Input(shape=input_shape)

    # Adjust the input feature dimensionality to 16 using a convolutional layer
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

    # Define the basic block
    def basic_block(x):
        branch = layers.Conv2D(16, (1, 1), activation='relu')(x)
        main_path = layers.Conv2D(16, (3, 3), activation='relu')(x)
        main_path = layers.BatchNormalization()(main_path)
        output = layers.Add()([main_path, branch])
        output = layers.Activation('relu')(output)
        return output

    # Create a three-level residual connection structure
    x = basic_block(x)  # First level

    # Second level: two residual blocks with an independent convolutional layer in each branch
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # Main path
    x = basic_block(x)  # Residual block 1
    branch = layers.Conv2D(32, (1, 1), activation='relu')(x)  # Branch 1
    x = layers.Add()([x, branch])
    x = basic_block(x)  # Residual block 2
    branch = layers.Conv2D(32, (1, 1), activation='relu')(x)  # Branch 2
    x = layers.Add()([x, branch])

    # Third level: extract features using a convolutional layer in the global branch
    branch = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # Global branch
    x = layers.Add()([x, branch])

    # Average pooling followed by a fully connected layer for classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model