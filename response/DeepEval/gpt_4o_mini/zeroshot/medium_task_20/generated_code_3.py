import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Path 1: 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 Convolution followed by two 3x3 Convolutions
    path2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
    path2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(path2)
    path2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by a single 3x3 Convolution
    path3 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
    path3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(path3)

    # Path 4: Max Pooling followed by a 1x1 Convolution
    path4 = layers.MaxPooling2D(pool_size=(2, 2))(input_layer)
    path4 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(path4)

    # Concatenate all paths
    concatenated = layers.concatenate([path1, path2, path3, path4])

    # Flatten the concatenated output
    flatten = layers.Flatten()(concatenated)

    # Fully connected layer
    dense_layer = layers.Dense(128, activation='relu')(flatten)

    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(dense_layer)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())