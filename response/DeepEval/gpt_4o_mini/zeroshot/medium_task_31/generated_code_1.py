import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    inputs = layers.Input(shape=(32, 32, 3))

    # Split the input along the channel dimension (3 channels)
    split_images = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Convolutional layers for each split
    conv1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(split_images[0])
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(split_images[1])
    conv3 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(split_images[2])

    # Concatenate the outputs along the channel dimension
    concatenated = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # Flatten the concatenated output
    flatten = layers.Flatten()(concatenated)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dense2 = layers.Dense(64, activation='relu')(dense1)

    # Output layer with 10 classes for CIFAR-10
    outputs = layers.Dense(10, activation='softmax')(dense2)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()