import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the MNIST images, which are 28x28 pixels and have a single channel
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # First convolutional layer followed by a max pooling layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional layer followed by another max pooling layer
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Third convolutional layer
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)

    # Flatten the feature maps into a one-dimensional vector
    x = layers.Flatten()(x)

    # First fully connected (dense) layer
    x = layers.Dense(64, activation='relu')(x)

    # Second fully connected (dense) layer
    x = layers.Dense(64, activation='relu')(x)

    # Output layer with 10 units (one for each class) and softmax activation
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model