import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for MNIST dataset images of shape (28, 28, 1)
    input_shape = (28, 28, 1)
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional layer with 32 kernels
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Block 1
    # Split the tensor into two groups along the last dimension
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(x)

    # First group operations
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(split[0])
    group1 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(group1)
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(group1)

    # Second group (no modification)
    group2 = split[1]

    # Merge both groups
    merged = layers.Concatenate()([group1, group2])

    # Block 2
    # Reshape input for channel shuffling
    height, width, channels = tf.shape(merged)[1], tf.shape(merged)[2], tf.shape(merged)[3]
    groups = 4  # Number of groups for channel shuffling
    channels_per_group = channels // groups
    
    # Reshape to (height, width, groups, channels_per_group)
    reshaped = layers.Reshape((height, width, groups, channels_per_group))(merged)

    # Permute the dimensions to shuffle channels
    permuted = layers.Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to original shape
    shuffled = layers.Reshape((height, width, channels))(permuted)

    # Flatten the output for classification
    flattened = layers.Flatten()(shuffled)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()  # Display the model architecture