from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))  

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch Path
    branch = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    branch = layers.MaxPooling2D((2, 2))(branch)

    # Combine paths
    x = layers.Add()([x, branch])

    # Flatten
    x = layers.Flatten()(x)

    # Fully Connected Layers
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model