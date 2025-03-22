import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for the MNIST dataset (28x28 images with 1 color channel)
    inputs = layers.Input(shape=(28, 28, 1))

    # First block with a main path and a branch path
    # Main path
    x_main = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x_main = layers.Conv2D(1, (3, 3), padding='same', activation='relu')(x_main)

    # Branch path
    x_branch = inputs

    # Combine both paths
    x_combined = layers.Add()([x_main, x_branch])

    # Second block with three max pooling layers
    # Max pooling layers with different sizes
    x_pool1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x_combined)
    x_pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_combined)
    x_pool3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x_combined)

    # Flatten the outputs of the pooling layers
    x_flat1 = layers.Flatten()(x_pool1)
    x_flat2 = layers.Flatten()(x_pool2)
    x_flat3 = layers.Flatten()(x_pool3)

    # Concatenate the flattened outputs
    x_concat = layers.Concatenate()([x_flat1, x_flat2, x_flat3])

    # Fully connected layers
    x_fc = layers.Dense(128, activation='relu')(x_concat)
    x_fc = layers.Dense(64, activation='relu')(x_fc)

    # Output layer with softmax activation for classification (10 classes for digits 0-9)
    outputs = layers.Dense(10, activation='softmax')(x_fc)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0  # Normalize and add channel dimension
x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()