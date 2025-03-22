from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Main Path
    x = layers.ReLU()(layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs))
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.ReLU()(layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x))
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch Path
    branch = layers.Conv2D(64, (1, 1), activation='relu')(inputs)

    # Concatenate paths
    x = layers.Add()([x, branch])

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model