from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # Main path
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Branch path
    y = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)

    # Concatenate main and branch paths
    outputs = layers.Add()([x, y])

    # Flatten and fully connected layer
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Model creation
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])