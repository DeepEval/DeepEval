import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First convolutional branch with 3x3 kernels
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)

    # Second convolutional branch with 5x5 kernels
    branch2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    branch2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(branch2)

    # Combine the branches using addition
    combined = layers.add([branch1, branch2])

    # Global average pooling layer
    pooled = layers.GlobalAveragePooling2D()(combined)

    # Fully connected layers with softmax attention weights
    dense1 = layers.Dense(128, activation='relu')(pooled)
    attention_weights = layers.Dense(10, activation='softmax')(dense1)

    # Multiply each branch's output by its corresponding attention weight
    branch1_output = layers.GlobalAveragePooling2D()(branch1)
    branch2_output = layers.GlobalAveragePooling2D()(branch2)

    weighted_branch1 = layers.multiply([branch1_output, attention_weights])
    weighted_branch2 = layers.multiply([branch2_output, attention_weights])

    # Combine weighted outputs
    final_output = layers.add([weighted_branch1, weighted_branch2])

    # Fully connected layer for final output
    output_layer = layers.Dense(10, activation='softmax')(final_output)

    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Optionally load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()