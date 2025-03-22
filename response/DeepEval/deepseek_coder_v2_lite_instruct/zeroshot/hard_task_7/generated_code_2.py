import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, SeparableConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # Block 1
    # Initial convolutional layer
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)

    # Split the input into two groups
    split = Lambda(lambda x: tf.split(x, 2, axis=-1))(x)
    group1, group2 = split[0], split[1]

    # Group 1 operations
    g1_conv1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(group1)
    g1_depthwise = SeparableConv2D(32, kernel_size=(3, 3), activation='relu')(g1_conv1)
    g1_conv2 = Conv2D(32, kernel_size=(1, 1), activation='relu')(g1_depthwise)

    # Group 2 remains unchanged

    # Merge the outputs from both groups
    merged = Concatenate(axis=-1)([g1_conv2, group2])

    # Block 2
    # Get the shape of the input
    shape = tf.keras.backend.int_shape(merged)
    height, width, channels = shape[1], shape[2], shape[3]

    # Reshape the input into four groups
    reshaped = Reshape((height, width, 2, channels // 2))(merged)

    # Swap the third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Flatten the input back to its original shape
    flattened = Flatten()(permuted)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()