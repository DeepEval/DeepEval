import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define a function to create the convolutional path for each split
    def conv_path(x):
        x = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)
        return x

    # Apply the convolutional path to each split
    group_outputs = [conv_path(split) for split in splits]

    # Combine the outputs from the three groups using addition
    combined = layers.add(group_outputs)

    # Fuse the combined features with the original input
    fused = layers.add([combined, input_layer])

    # Flatten the features
    flattened = layers.Flatten()(fused)

    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.summary()