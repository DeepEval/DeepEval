import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Path 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)

    # Path 2
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.AveragePooling2D((2, 2))(x2)

    # Combine Path 1 and Path 2
    x = layers.Add()([x1, x2])

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_shape, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()