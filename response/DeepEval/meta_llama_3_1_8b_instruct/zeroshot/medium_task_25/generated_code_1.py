# Import necessary libraries
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape for CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Path 1: Single 1x1 convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path1 = layers.Flatten()(path1)

    # Path 2: Average pooling followed by 1x1 convolution
    path2 = layers.AveragePooling2D((2, 2))(inputs)
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(path2)
    path2 = layers.Flatten()(path2)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path3 = layers.Concatenate()([layers.Conv2D(32, (1, 3), activation='relu')(path3),
                                  layers.Conv2D(32, (3, 1), activation='relu')(path3)])
    path3 = layers.Flatten()(path3)

    # Path 4: 1x1 convolution followed by 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path4 = layers.Conv2D(32, (3, 3), activation='relu')(path4)
    path4 = layers.Concatenate()([layers.Conv2D(32, (1, 3), activation='relu')(path4),
                                  layers.Conv2D(32, (3, 1), activation='relu')(path4)])
    path4 = layers.Flatten()(path4)

    # Concatenate the outputs of the four paths
    x = layers.Concatenate()([path1, path2, path3, path4])

    # Add a fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model