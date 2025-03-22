import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Flatten, Lambda, Concatenate, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape for MNIST (28x28 grayscale images)
    input_shape = (28, 28, 1)
    inputs = Input(shape=input_shape)

    # Initial convolutional layer with 32 filters
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Block 1
    # Split the input into two groups along the last dimension
    def split_function(x):
        return tf.split(x, num_or_size_splits=2, axis=-1)

    split = Lambda(split_function)(x)

    # First group operations
    group1 = Conv2D(16, (1, 1), activation='relu')(split[0])
    group1 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(group1)
    group1 = Conv2D(16, (1, 1), activation='relu')(group1)

    # Second group remains unchanged
    group2 = split[1]

    # Concatenate the outputs of both groups
    block1_output = Concatenate(axis=-1)([group1, group2])

    # Block 2
    # Get the shape of the input
    input_shape = tf.shape(block1_output)
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]

    # Determine the number of groups
    groups = 4
    channels_per_group = channels // groups

    # Reshape input to (height, width, groups, channels_per_group)
    reshaped = Reshape((height, width, groups, channels_per_group))(block1_output)

    # Permute dimensions to (height, width, channels_per_group, groups)
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to the original shape
    block2_output = Reshape((height, width, channels))(permuted)

    # Flatten the output and add a fully connected layer for classification
    x = Flatten()(block2_output)
    outputs = Dense(10, activation='softmax')(x)  # MNIST has 10 classes

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
# Summary of the model architecture
model.summary()