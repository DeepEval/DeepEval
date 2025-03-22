# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Split the input into three groups along the last dimension
    inputs = keras.Input(shape=input_shape)
    split = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Define the feature extraction layers for each group
    group1 = layers.SeparableConv2D(32, (1, 1), activation='relu', input_shape=input_shape)(split[0])
    group2 = layers.SeparableConv2D(32, (3, 3), activation='relu')(split[1])
    group3 = layers.SeparableConv2D(32, (5, 5), activation='relu')(split[2])

    # Concatenate and fuse the outputs of the three groups
    fused = layers.Concatenate()([group1, group2, group3])

    # Apply a depthwise separable convolution layer to the fused features
    fused = layers.SeparableConv2D(32, (3, 3), activation='relu')(fused)

    # Flatten the fused features into a one-dimensional vector
    flat = layers.Flatten()(fused)

    # Define the fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flat)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Call the function to create the model
model = dl_model()