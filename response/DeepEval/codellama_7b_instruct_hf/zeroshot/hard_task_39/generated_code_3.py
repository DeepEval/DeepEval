import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the input shape and the output size
input_shape = (28, 28, 1)
num_classes = 10

# Define the first specialized block
block1 = layers.Sequential([
    layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2)),
    layers.MaxPooling2D(pool_size=(8, 8), strides=(4, 4)),
    layers.Flatten()
])

# Define the second specialized block
block2 = layers.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    layers.Flatten()
])

# Define the fully connected layer and the reshape operation
fc = layers.Dense(128, activation='relu')
reshape = layers.Reshape((-1, 128))

# Define the model
model = models.Sequential()
model.add(block1)
model.add(fc)
model.add(reshape)
model.add(block2)
model.add(layers.Flatten())
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model with the Adam optimizer and the categorical cross-entropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Return the constructed model
return model