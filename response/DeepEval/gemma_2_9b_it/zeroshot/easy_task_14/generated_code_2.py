from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers to learn channel correlations
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(3 * 32, activation='relu')(x)

    # Reshape to align with input shape and multiply element-wise
    x = layers.Reshape((32, 32, 3))(x)
    x = layers.multiply([x, input_tensor])

    # Flatten and final fully connected layer
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=x)

    return model