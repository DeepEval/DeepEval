import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Lambda, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1
    # Initial convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Split the input into two groups
    split_index = tf.shape(x)[-1] // 2
    split = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=2, axis=-1))(x)

    # First group (Group 1)
    group1 = split[0]
    group1 = Conv2D(32, (1, 1), activation='relu')(group1)
    group1 = DepthwiseConv2D((3, 3), activation='relu')(group1)
    group1 = Conv2D(32, (1, 1), activation='relu')(group1)

    # Second group (Group 2)
    group2 = split[1]

    # Concatenate the outputs from both groups
    merged = Concatenate()([group1, group2])

    # Block 2
    # Get the shape of the input
    shape = tf.shape(merged)

    # Reshape the input into four groups
    reshaped = Reshape((shape[1], shape[2], 2, shape[3] // 2))(merged)

    # Swap the third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Flatten the input
    flattened = Flatten()(permuted)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()