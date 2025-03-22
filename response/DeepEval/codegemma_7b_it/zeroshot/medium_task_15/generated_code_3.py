from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, concatenate, Conv2D, AveragePooling2D
from tensorflow.keras import Sequential

def dl_model():
    model = Sequential([
        Conv2D(filters=256, kernel_size=(5,5), padding="same", input_shape=(32, 32, 3)),
        BatchNormalization(),
        ReLU(),
        GlobalAveragePooling2D(),
        Dense(units=256),
        Dense(units=256),
        Reshape((32, 32, 1)),
        Multiply(),
        concatenate([Reshape((32, 32, 3))]),
        Conv2D(filters=256, kernel_size=(1,1), padding="same"),
        AveragePooling2D(),
        Dense(units=10)
    ])

    return model