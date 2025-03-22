import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # First block: main path and branch path
    # Main path
    x_main = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x_main = layers.Dropout(0.2)(x_main)
    x_main = layers.Conv2D(3, (3, 3), padding='same')(x_main)  # Restore channels to 3
    # Branch path
    x_branch = inputs

    # Combine paths
    x_combined = layers.add([x_main, x_branch])

    # Second block: split into three groups with separable convolutions
    x_split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x_combined)

    # Group 1: 1x1 separable convolution
    x1 = layers.SeparableConv2D(32, (1, 1), padding='same')(x_split[0])
    x1 = layers.Dropout(0.2)(x1)

    # Group 2: 3x3 separable convolution
    x2 = layers.SeparableConv2D(32, (3, 3), padding='same')(x_split[1])
    x2 = layers.Dropout(0.2)(x2)

    # Group 3: 5x5 separable convolution
    x3 = layers.SeparableConv2D(32, (5, 5), padding='same')(x_split[2])
    x3 = layers.Dropout(0.2)(x3)

    # Concatenate the outputs from the three groups
    x_concat = layers.Concatenate(axis=-1)([x1, x2, x3])

    # Flatten and fully connected layer for predictions
    x_flattened = layers.Flatten()(x_concat)
    outputs = layers.Dense(10, activation='softmax')(x_flattened)  # 10 classes for CIFAR-10

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()  # Display the model architecture