from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ]
    )
    return model