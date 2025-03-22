# Import necessary libraries
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    The model features a specialized block designed to capture local features through a 3x3 convolutional layer,
    followed by two 1x1 convolutional layers for additional processing. An average pooling layer is then applied
    to downsample the feature maps, and a dropout layer is included to mitigate overfitting.
    After two consecutive the blocks, a global average pooling layer converts the high-dimensional feature maps
    into a one-dimensional vector. This is followed by a flattening layer and a fully connected layer that produces
    the final classification output.
    
    Returns:
    model: A constructed deep learning model.
    """

    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the first block
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Define the second block
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Apply global average pooling to downsample the feature maps
    x = layers.GlobalAveragePooling2D()(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # Define the output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Construct the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
print(model.summary())