import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    y = layers.GlobalAveragePooling2D()(input_layer)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(32, activation='relu')(y)
    y = layers.Dense(3 * 3 * 128, activation='sigmoid')(y)  # Output of shape (3*3*128) for channel weights
    y = layers.Reshape((3, 3, 128))(y)  # Reshape to match the dimensions of the feature maps

    # Multiply the inputs from the main path with the branch path weights
    z = layers.Multiply()([x, y])

    # Adding additional fully connected layers for classification
    z = layers.GlobalAveragePooling2D()(z)  # Global pooling to flatten the output
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dense(10, activation='softmax')(z)  # 10 classes for CIFAR-10

    # Constructing the model
    model = models.Model(inputs=input_layer, outputs=z)

    return model

# Example usage
model = dl_model()
model.summary()