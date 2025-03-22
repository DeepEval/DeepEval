import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_tensor = layers.Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    
    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)

    # Third convolutional layer
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

    # Direct processing of the input through a separate convolutional layer
    direct_conv = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_tensor)

    # Adding the outputs of the first two convolutional layers with the output of the third layer
    added_output = layers.add([conv1, conv2, conv3])

    # Adding the direct convolution output to the added output
    combined_output = layers.add([added_output, direct_conv])

    # Flatten the combined output for the fully connected layers
    flatten = layers.Flatten()(combined_output)

    # Fully connected layer 1
    dense1 = layers.Dense(256, activation='relu')(flatten)

    # Fully connected layer 2
    dense2 = layers.Dense(128, activation='relu')(dense1)

    # Output layer for classification (10 classes for CIFAR-10)
    output = layers.Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = models.Model(inputs=input_tensor, outputs=output)

    return model

# Example usage:
model = dl_model()
model.summary()