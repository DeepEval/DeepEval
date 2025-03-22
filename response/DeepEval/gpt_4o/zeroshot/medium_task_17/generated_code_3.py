import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Permute, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # CIFAR-10 images are 32x32 with 3 channels
    input_shape = (32, 32, 3)
    num_classes = 10

    # Define input layer
    inputs = Input(shape=input_shape)

    # Reshape the input tensor to (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = input_shape[2] // groups
    reshaped = Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(inputs)

    # Permute to swap the third and fourth dimensions for channel shuffling
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to the original input shape
    reshaped_back = Reshape(input_shape)(permuted)

    # Flatten the output for the fully connected layer
    flattened = Flatten()(reshaped_back)

    # Fully connected layer with softmax activation for classification
    outputs = Dense(num_classes, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()