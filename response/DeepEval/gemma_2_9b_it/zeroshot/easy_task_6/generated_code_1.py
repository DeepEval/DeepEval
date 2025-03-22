from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_layer = keras.Input(shape=(28, 28, 1))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # Branch path
    branch = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(input_layer)

    # Combine paths
    x = layers.Add()([x, branch])

    # Flatten and fully connected
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model