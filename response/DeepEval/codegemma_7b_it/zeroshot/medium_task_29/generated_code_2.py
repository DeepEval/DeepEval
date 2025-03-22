from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(1, 1), strides=1),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        MaxPooling2D(pool_size=(4, 4), strides=4),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model