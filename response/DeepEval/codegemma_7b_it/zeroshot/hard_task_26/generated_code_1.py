import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    input_img = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Branch 1
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(x)
    branch1 = layers.BatchNormalization()(branch1)

    # Branch 2
    branch2 = layers.MaxPooling2D()(x)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.UpSampling2D()(branch2)

    # Branch 3
    branch3 = layers.MaxPooling2D()(x)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.UpSampling2D()(branch3)
    branch3 = layers.UpSampling2D()(branch3)

    # Concatenate branches and main path
    concat = layers.concatenate([x, branch1, branch2, branch3])

    # Main path continuation
    x = layers.Conv2D(64, (1, 1), activation='relu')(concat)
    x = layers.BatchNormalization()(x)

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Model definition
    model = keras.Model(inputs=input_img, outputs=x)

    return model