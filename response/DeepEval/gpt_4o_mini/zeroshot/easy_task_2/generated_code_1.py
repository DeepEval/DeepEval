import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(224, 224, 3))

    # First feature extraction block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    # Second feature extraction block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    # Additional convolutional layers
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    # Flattening the feature maps
    x = layers.Flatten()(x)

    # Fully connected layers with Dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer with softmax activation for 1000 classes
    output_layer = layers.Dense(1000, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()