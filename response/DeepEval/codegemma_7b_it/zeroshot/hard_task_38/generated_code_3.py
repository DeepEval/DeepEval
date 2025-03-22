from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

def dl_model():

    inputs = keras.Input(shape=(28, 28, 1))

    # Pathway 1
    pathway1 = inputs
    for _ in range(3):
        pathway1 = layers.BatchNormalization()(pathway1)
        pathway1 = layers.Activation('relu')(pathway1)
        pathway1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(pathway1)

    # Pathway 2
    pathway2 = inputs
    for _ in range(3):
        pathway2 = layers.BatchNormalization()(pathway2)
        pathway2 = layers.Activation('relu')(pathway2)
        pathway2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(pathway2)

    # Concatenate outputs from both pathways
    concat = layers.concatenate([pathway1, pathway2])

    # Fully connected layers
    x = layers.BatchNormalization()(concat)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=10, activation='softmax')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=x)

    return model