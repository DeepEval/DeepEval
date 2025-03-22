from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            layers.AveragePooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.AveragePooling2D(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.AveragePooling2D(),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1000, activation="softmax"),
        ]
    )
    return model