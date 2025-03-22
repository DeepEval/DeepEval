from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential(
        [
            layers.AvgPool2D(pool_size=(5, 5), strides=(3, 3), input_shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu'),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(units=10, activation='softmax')
        ]
    )
    return model