import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 1x1 Convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch1 = layers.Dropout(0.5)(branch1)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.Dropout(0.5)(branch2)

    # Branch 3: 1x1 Convolution followed by two consecutive 3x3 Convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.Dropout(0.5)(branch3)

    # Branch 4: Average Pooling followed by 1x1 Convolution
    branch4 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)
    branch4 = layers.Dropout(0.5)(branch4)

    # Concatenate branches
    concatenated = layers.concatenate([branch1, branch2, branch3, branch4])

    # Flatten and fully connected layers
    flattened = layers.Flatten()(concatenated)
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    dense2 = layers.Dropout(0.5)(dense2)
    outputs = layers.Dense(10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Example usage:
# model = dl_model()
# model.summary()