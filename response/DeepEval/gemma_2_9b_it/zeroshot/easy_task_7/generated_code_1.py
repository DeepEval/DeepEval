from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)

    # Branch Path
    branch_x = layers.Conv2D(64, (1, 1), activation='relu')(input_tensor)

    # Combine paths
    x = layers.Add()([x, branch_x])

    # Flatten and classify
    x = layers.Flatten()(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model