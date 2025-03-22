from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch path
    branch_inputs = layers.GlobalAveragePooling2D()(inputs)
    branch_x = layers.Dense(128, activation='relu')(branch_inputs)
    branch_x = layers.Dense(128, activation='relu')(branch_x)
    branch_weights = layers.Reshape((32, 32, 128))(branch_x)

    # Multiply channel weights with input
    x = layers.multiply([x, branch_weights])

    # Combine paths and classify
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model