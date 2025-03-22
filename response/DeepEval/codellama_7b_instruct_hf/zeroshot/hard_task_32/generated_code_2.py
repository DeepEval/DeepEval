import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the depthwise separable convolutional layer block
    separable_conv_block = keras.Sequential([
        layers.SeparableConv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.SeparableConv2D(32, 3, activation='relu')
    ])

    # Define the 1x1 convolutional layer block
    conv_block = keras.Sequential([
        layers.Conv2D(32, 1, activation='relu'),
        layers.Conv2D(32, 1, activation='relu')
    ])

    # Define the concatenation layer for the three branches
    concat_layer = layers.Concatenate()

    # Define the dropout layers
    dropout_layer1 = layers.Dropout(0.2)
    dropout_layer2 = layers.Dropout(0.2)

    # Define the fully connected layers
    dense_layer1 = layers.Dense(64, activation='relu')
    dense_layer2 = layers.Dense(10, activation='softmax')

    # Define the model
    model = keras.Sequential([
        # Branch 1
        separable_conv_block,
        dropout_layer1,
        conv_block,
        dropout_layer2,
        concat_layer,
        # Branch 2
        separable_conv_block,
        dropout_layer1,
        conv_block,
        dropout_layer2,
        concat_layer,
        # Branch 3
        separable_conv_block,
        dropout_layer1,
        conv_block,
        dropout_layer2,
        concat_layer,
        # Fully connected layers
        dense_layer1,
        dropout_layer1,
        dense_layer2
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model