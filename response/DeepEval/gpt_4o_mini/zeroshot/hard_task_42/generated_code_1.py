import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for 28x28 grayscale images
    input_shape = (28, 28, 1)
    inputs = layers.Input(shape=input_shape)

    # Block 1: Three parallel paths with max pooling of different scales
    # Path 1: Max pooling (1x1)
    path1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path1 = layers.Flatten()(path1)
    path1 = layers.Dropout(0.5)(path1)

    # Path 2: Max pooling (2x2)
    path2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path2 = layers.Flatten()(path2)
    path2 = layers.Dropout(0.5)(path2)

    # Path 3: Max pooling (4x4)
    path3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    path3 = layers.Flatten()(path3)
    path3 = layers.Dropout(0.5)(path3)

    # Concatenate the outputs of the three paths
    block1_output = layers.concatenate([path1, path2, path3])

    # Fully connected layer after Block 1
    fc1 = layers.Dense(128, activation='relu')(block1_output)
    reshaped_output = layers.Reshape((4, 4, 8))(fc1)  # Reshape to a 4D tensor (assuming 8 channels)

    # Block 2: Four parallel paths
    # Path 1: 1x1 Convolution
    path1_b2 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped_output)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2_b2 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped_output)
    path2_b2 = layers.Conv2D(32, (1, 7), activation='relu')(path2_b2)
    path2_b2 = layers.Conv2D(32, (7, 1), activation='relu')(path2_b2)

    # Path 3: Alternating 7x1 and 1x7 Convolutions
    path3_b2 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped_output)
    path3_b2 = layers.Conv2D(32, (7, 1), activation='relu')(path3_b2)
    path3_b2 = layers.Conv2D(32, (1, 7), activation='relu')(path3_b2)
    path3_b2 = layers.Conv2D(32, (7, 1), activation='relu')(path3_b2)
    path3_b2 = layers.Conv2D(32, (1, 7), activation='relu')(path3_b2)

    # Path 4: Average pooling with 1x1 Convolution
    path4_b2 = layers.AveragePooling2D(pool_size=(2, 2))(reshaped_output)
    path4_b2 = layers.Conv2D(32, (1, 1), activation='relu')(path4_b2)

    # Concatenate the outputs of the four paths
    block2_output = layers.concatenate([path1_b2, path2_b2, path3_b2, path4_b2])

    # Flatten the output from Block 2
    flattened_output = layers.Flatten()(block2_output)

    # Fully connected layers for final classification
    fc2 = layers.Dense(128, activation='relu')(flattened_output)
    outputs = layers.Dense(10, activation='softmax')(fc2)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()