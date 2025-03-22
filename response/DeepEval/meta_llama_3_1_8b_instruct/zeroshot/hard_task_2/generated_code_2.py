# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    # using tf.split within a Lambda layer
    split_inputs = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Define the main path
    main_path = []
    for i in range(3):
        # Extract features from each group using a series of convolutions
        x = split_inputs[i]
        x = layers.Conv2D(6, (1, 1), activation='relu')(x)  # 1x1 convolution
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)  # 3x3 convolution
        x = layers.Conv2D(6, (1, 1))(x)  # 1x1 convolution
        main_path.append(x)

    # Combine the outputs from the three groups
    # using an addition operation
    combined_main_path = layers.Add()(main_path)

    # Fuse the main path with the original input layer
    # through another addition
    x = layers.Add()([combined_main_path, inputs])

    # Flatten the combined features into a one-dimensional vector
    x = layers.Flatten()(x)

    # Create a fully connected layer for classification
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model