from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Path 1: Three sequential convolutional layers
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.Conv2D(128, (3, 3), activation='relu')(x1)

    # Path 2: Direct convolutional layer
    x2 = layers.Conv2D(128, (3, 3), activation='relu')(input_tensor)

    # Combine outputs
    x = layers.Add()([x1, x2])

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)  

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model