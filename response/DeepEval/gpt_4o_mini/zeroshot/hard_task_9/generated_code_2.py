import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 1x1 Convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)

    # Concatenate the outputs of the three branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3])

    # 1x1 Convolution to adjust the output dimensions
    output = layers.Conv2D(3, (1, 1), activation='relu')(concatenated)

    # Adding the branch directly connected to the input
    added = layers.Add()([output, inputs])

    # Flatten the output to feed into the fully connected layers
    flatten = layers.Flatten()(added)

    # Fully connected layers for classification
    dense1 = layers.Dense(512, activation='relu')(flatten)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    outputs = layers.Dense(10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()