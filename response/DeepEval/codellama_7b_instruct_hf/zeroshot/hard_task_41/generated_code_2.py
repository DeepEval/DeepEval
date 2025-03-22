import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    branch1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch1)
    branch3 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(branch2)
    branch4 = layers.AveragePooling2D(pool_size=(2, 2))(branch3)
    branch5 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(branch4)
    branch6 = layers.AveragePooling2D(pool_size=(2, 2))(branch5)
    branch7 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch6)
    branch8 = layers.AveragePooling2D(pool_size=(2, 2))(branch7)
    branch9 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(branch8)
    branch10 = layers.AveragePooling2D(pool_size=(2, 2))(branch9)
    x = layers.Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8, branch9, branch10])
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    branch11 = layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    branch12 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch11)
    branch13 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch12)
    branch14 = layers.AveragePooling2D(pool_size=(2, 2))(branch13)
    branch15 = layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(branch14)
    branch16 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch15)
    branch17 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch16)
    branch18 = layers.AveragePooling2D(pool_size=(2, 2))(branch17)
    branch19 = layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(branch18)
    branch20 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch19)
    branch21 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch20)
    branch22 = layers.AveragePooling2D(pool_size=(2, 2))(branch21)
    x = layers.Concatenate()([branch11, branch12, branch13, branch14, branch15, branch16, branch17, branch18, branch19, branch20, branch21, branch22])
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='dl_model')

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model