from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # Branch Path
    branch_x = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(input_tensor)

    # Combination
    x = layers.Add()([x, branch_x])

    # Flatten and Fully Connected
    x = layers.Flatten()(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model