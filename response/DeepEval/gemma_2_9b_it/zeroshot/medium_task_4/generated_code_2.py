from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Path 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x1 = layers.Conv2D(128, (3, 3), activation='relu')(x1)
    x1 = layers.Conv2D(256, (3, 3), activation='relu')(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)

    # Path 2
    x2 = layers.Conv2D(128, (3, 3), activation='relu')(input_tensor)

    # Concatenate features from both paths
    x = layers.Concatenate()([x1, x2])

    # Flatten and classify
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model