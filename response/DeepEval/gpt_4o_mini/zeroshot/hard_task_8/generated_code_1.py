import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))

    # Block 1 - Main Path
    x1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    x1 = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x1)

    # Block 1 - Branch Path
    x2 = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x2)

    # Concatenate both paths
    x = layers.concatenate([x1, x2], axis=-1)

    # Block 2 - Reshape and Channel Shuffle
    shape_before_reshape = tf.shape(x)
    height, width = shape_before_reshape[1], shape_before_reshape[2]
    groups = 4  # You can change this if needed
    channels_per_group = shape_before_reshape[3] // groups

    # Reshape to (height, width, groups, channels_per_group)
    x = layers.Reshape((height, width, groups, channels_per_group))(x)

    # Permute to (height, width, channels_per_group, groups)
    x = layers.Permute((0, 1, 3, 2))(x)

    # Reshape back to original shape (height, width, channels)
    x = layers.Reshape((height, width, -1))(x)

    # Final fully connected layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct model
    model = models.Model(inputs, outputs)

    return model

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)