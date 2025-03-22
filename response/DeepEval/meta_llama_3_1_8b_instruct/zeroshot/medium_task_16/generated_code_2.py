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
    groups = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Apply 1x1 convolutions to each group independently
    convs = []
    for group in groups:
        convs.append(layers.Conv2D(
            filters=int(input_shape[-1] / 3),
            kernel_size=1,
            activation='relu'
        )(group))

    # Downsample each group via average pooling
    downsampled = []
    for conv in convs:
        downsampled.append(layers.AveragePooling2D(pool_size=(2, 2))(conv))

    # Concatenate the three groups of feature maps along the channel dimension
    concatenated = layers.Concatenate()(downsampled)

    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = layers.Flatten()(concatenated)

    # Create two fully connected layers for classification
    x = layers.Dense(64, activation='relu')(flattened)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create and print the model summary
model = dl_model()
model.summary()