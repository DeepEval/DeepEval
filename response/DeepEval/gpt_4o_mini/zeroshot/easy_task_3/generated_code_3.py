import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for 28x28 grayscale images (MNIST)
    inputs = layers.Input(shape=(28, 28, 1))

    # First block: Conv -> Conv -> Max Pooling
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second block: Conv -> Conv -> Conv -> Max Pooling
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)  # Output layer for 10 classes (digits 0-9)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=x)

    # Compile the model (optional, for training)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()

# Summary of the model
model.summary()