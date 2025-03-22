import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for images of shape 224x224x3
    input_layer = layers.Input(shape=(224, 224, 3))

    # First feature extraction block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.AveragePooling2D((2, 2))(x)

    # Second feature extraction block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Additional convolutional layers
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Final average pooling layer
    x = layers.AveragePooling2D((2, 2))(x)

    # Flattening the feature maps
    x = layers.Flatten()(x)

    # First fully connected layer with dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Second fully connected layer with dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer with softmax for 1000 classes
    output_layer = layers.Dense(1000, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.summary()  # To print the model summary