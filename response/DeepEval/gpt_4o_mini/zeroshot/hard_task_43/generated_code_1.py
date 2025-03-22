import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # Block 1: Parallel paths with average pooling layers
    path1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flattening each path
    flat_path1 = layers.Flatten()(path1)
    flat_path2 = layers.Flatten()(path2)
    flat_path3 = layers.Flatten()(path3)

    # Concatenating the flattened outputs
    concat_block1 = layers.concatenate([flat_path1, flat_path2, flat_path3])

    # Fully connected layer
    fc1 = layers.Dense(128, activation='relu')(concat_block1)

    # Reshape to 4D tensor for Block 2
    reshape_layer = layers.Reshape((1, 1, 128))(fc1)

    # Block 2: Branches for feature extraction
    # Branch 1: 1x1, 3x3 convolution
    branch1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(reshape_layer)
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)

    # Branch 2: 1x1, 1x7, 7x1, 3x3 convolutions
    branch2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = layers.Conv2D(32, (1, 7), padding='same', activation='relu')(branch2)
    branch2 = layers.Conv2D(32, (7, 1), padding='same', activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Average pooling
    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(reshape_layer)

    # Concatenating the outputs from all branches
    concat_block2 = layers.concatenate([branch1, branch2, branch3])

    # Flattening the concatenated output
    flat_block2 = layers.Flatten()(concat_block2)

    # Fully connected layers for classification
    fc2 = layers.Dense(128, activation='relu')(flat_block2)
    output_layer = layers.Dense(10, activation='softmax')(fc2)  # MNIST has 10 classes

    # Constructing the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()