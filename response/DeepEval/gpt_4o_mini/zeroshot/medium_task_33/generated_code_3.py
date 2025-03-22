import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the model
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    channel_groups = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define separable convolution layers for each channel group
    conv1_group1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(channel_groups[0])
    conv1_group2 = layers.SeparableConv2D(32, (3, 3), activation='relu')(channel_groups[1])
    conv1_group3 = layers.SeparableConv2D(32, (5, 5), activation='relu')(channel_groups[2])

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([conv1_group1, conv1_group2, conv1_group3])

    # Add more convolutional layers if needed (optional)
    x = layers.SeparableConv2D(64, (3, 3), activation='relu')(concatenated)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output before fully connected layers
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output layer with softmax activation for multi-class classification
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()