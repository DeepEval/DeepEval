import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    path1 = keras.Sequential([
        layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'),
        layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')
    ])(inputs)

    path2 = keras.Sequential([
        layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
    ])(inputs)

    concat = keras.layers.Add()([path1, path2])

    flatten = keras.layers.Flatten()(concat)

    output = keras.layers.Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=inputs, outputs=output)

    return model


if __name__ == '__main__':
    model = dl_model()
    print(model.summary())