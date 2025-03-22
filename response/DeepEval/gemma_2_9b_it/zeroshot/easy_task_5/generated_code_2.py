from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
  
    model = keras.Sequential(
        [
            layers.Conv2D(filters=1, kernel_size=(1, 1), activation="relu", input_shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(filters=1, kernel_size=(1, 1), activation="relu"),
            layers.Flatten(),
            layers.Dense(units=10, activation="softmax"),
        ]
    )
    return model