# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model features two processing pathways, each employing a repeated block structure executed three times.
    The block includes batch normalization and ReLU activation, followed by a 3x3 convolutional layer that extracts features while preserving spatial dimensions.
    The original input of the block is then concatenated with the new features along the channel dimension.
    The outputs from both pathways are merged through concatenation and classified using two fully connected layers.
    """

    # Define the input shape of the MNIST dataset
    input_shape = (28, 28, 1)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the first processing pathway
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Repeat the block structure three times for the first pathway
    for _ in range(3):
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, layers.Conv2D(32, (3, 3), padding='same')(x)])
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Define the second processing pathway
    y = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    # Repeat the block structure three times for the second pathway
    for _ in range(3):
        y = layers.Conv2D(32, (3, 3), padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Concatenate()([y, layers.Conv2D(32, (3, 3), padding='same')(y)])
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)

    # Merge the outputs from both pathways through concatenation
    merged = layers.Concatenate()([x, y])

    # Flatten the merged output
    merged = layers.Flatten()(merged)

    # Define the first fully connected layer
    merged = layers.Dense(64, activation='relu')(merged)

    # Define the second fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(merged)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Test the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()