import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 channels

    # Block 1: Splitting the input into three groups
    split_channels = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Convolutional layers with varying kernel sizes
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(split_channels[0])
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(split_channels[1])
    conv3 = layers.Conv2D(32, (5, 5), activation='relu')(split_channels[2])

    # Dropout layer
    dropped = layers.Dropout(0.5)(layers.concatenate([conv1, conv2, conv3]))

    # Block 2: Four branches
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(dropped)

    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(dropped)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)

    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(dropped)
    branch3 = layers.Conv2D(32, (5, 5), activation='relu')(branch3)

    branch4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropped)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenating the outputs of the branches
    fusion = layers.concatenate([branch1, branch2, branch3, branch4])

    # Flattening and fully connected layer
    flatten = layers.Flatten()(fusion)
    outputs = layers.Dense(10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Creating the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()