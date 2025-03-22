import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply different convolutions to each split
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(split[0])  # 1x1 convolution
    conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(split[1])  # 3x3 convolution
    conv3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(split[2])  # 5x5 convolution

    # Concatenate the outputs from the convolutions
    concatenated = layers.concatenate([conv1, conv2, conv3], axis=-1)

    # Flatten the concatenated features
    flattened = layers.Flatten()(concatenated)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(64, activation='relu')(dense1)

    # Output layer with softmax activation for classification (10 classes)
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()