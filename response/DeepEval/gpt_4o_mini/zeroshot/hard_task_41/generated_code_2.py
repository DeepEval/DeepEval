import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels and single channel
    inputs = layers.Input(shape=input_shape)

    # Block 1: Three parallel paths with average pooling layers of different scales
    path1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten and Dropout
    path1_flat = layers.Flatten()(path1)
    path2_flat = layers.Flatten()(path2)
    path3_flat = layers.Flatten()(path3)

    path1_dropout = layers.Dropout(0.5)(path1_flat)
    path2_dropout = layers.Dropout(0.5)(path2_flat)
    path3_dropout = layers.Dropout(0.5)(path3_flat)

    # Concatenating the outputs from the three paths
    block1_output = layers.Concatenate()([path1_dropout, path2_dropout, path3_dropout])

    # Fully connected layer before reshaping
    fc1 = layers.Dense(128, activation='relu')(block1_output)

    # Reshape into 4D tensor
    reshaped_output = layers.Reshape((1, 1, 128))(fc1)

    # Block 2: Multiple branch connections for feature extraction
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped_output)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(reshaped_output)
    
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped_output)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)

    branch4 = layers.AveragePooling2D(pool_size=(2, 2))(reshaped_output)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs from the branches
    block2_output = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Flattening the concatenated output
    block2_flat = layers.Flatten()(block2_output)

    # Fully connected layers for classification
    fc2 = layers.Dense(128, activation='relu')(block2_flat)
    outputs = layers.Dense(10, activation='softmax')(fc2)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()