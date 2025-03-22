import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = layers.Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    # Concatenating input with output of first layer
    concat1 = layers.Concatenate()([input_layer, conv1])

    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(concat1)
    # Concatenating input with output of second layer
    concat2 = layers.Concatenate()([concat1, conv2])

    # Third convolutional layer
    conv3 = layers.Conv2D(128, (3, 3), activation='relu')(concat2)

    # Concatenating input with output of third layer
    concat3 = layers.Concatenate()([concat2, conv3])

    # Flattening the output
    flatten = layers.Flatten()(concat3)

    # Fully connected layer 1
    fc1 = layers.Dense(256, activation='relu')(flatten)
    
    # Fully connected layer 2
    fc2 = layers.Dense(128, activation='relu')(fc1)

    # Output layer for classification (CIFAR-10 has 10 classes)
    output_layer = layers.Dense(10, activation='softmax')(fc2)

    # Constructing the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model