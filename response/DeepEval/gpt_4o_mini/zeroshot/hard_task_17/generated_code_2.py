import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB
    inputs = layers.Input(shape=input_shape)

    # Block 1
    # Global Average Pooling layer
    gap = layers.GlobalAveragePooling2D()(inputs)
    
    # Fully connected layers in Block 1
    fc1 = layers.Dense(units=32, activation='relu')(gap)
    fc2 = layers.Dense(units=32, activation='relu')(fc1)
    
    # Reshape and multiply with input
    reshaped_weights = layers.Reshape((1, 1, 32))(fc2)
    weighted_output = layers.multiply([inputs, reshaped_weights])

    # Block 2
    # Convolutional layers
    conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(weighted_output)
    conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch connection from Block 1 to Block 2
    branch = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(weighted_output)

    # Add outputs of Block 2 and the branch
    combined = layers.add([pool, branch])

    # Flatten combined output
    flattened = layers.Flatten()(combined)

    # Fully connected layers for classification
    fc3 = layers.Dense(units=64, activation='relu')(flattened)
    outputs = layers.Dense(units=10, activation='softmax')(fc3)  # CIFAR-10 has 10 classes

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Print the model summary