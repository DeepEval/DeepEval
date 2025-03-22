# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the initial input
    inputs = keras.Input(shape=input_shape, name='inputs')

    # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    x = layers.Conv2D(3, (1, 1), activation='relu', name='conv_1x1')(inputs)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = layers.DepthwiseConv2D((3, 3), activation='relu', name='conv_3x3_depthwise')(x)

    # Compute channel attention weights through global average pooling followed by two fully connected layers
    channel_attention = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    channel_attention = layers.Dense(128, activation='relu', name='dense_1')(channel_attention)
    channel_attention = layers.Dense(3, activation='sigmoid', name='dense_2')(channel_attention)
    channel_attention = layers.Reshape((1, 1, 3))(channel_attention)

    # Multiply the initial features with the channel attention weights
    x = layers.Multiply(name='multiply')([x, channel_attention])

    # Reduce the dimensionality with a 1x1 convolution
    x = layers.Conv2D(3, (1, 1), name='conv_1x1_output')(x)

    # Combine the output with the initial input
    x = layers.Add(name='add')([x, inputs])

    # Pass the output through a flattening layer and a fully connected layer for classification
    x = layers.Flatten(name='flatten')(x)
    outputs = layers.Dense(10, activation='softmax', name='dense_output')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()