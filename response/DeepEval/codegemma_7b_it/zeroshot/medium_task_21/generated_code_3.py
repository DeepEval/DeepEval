import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    branch1 = layers.Conv2D(filters=64, kernel_size=1, activation='relu')(inputs)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = layers.Conv2D(filters=64, kernel_size=1, activation='relu')(inputs)
    branch2 = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 Convolution followed by Two Consecutive 3x3 Convolutions
    branch3 = layers.Conv2D(filters=64, kernel_size=1, activation='relu')(inputs)
    branch3 = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(branch3)

    # Branch 4: Average Pooling followed by 1x1 Convolution
    branch4 = layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    branch4 = layers.Conv2D(filters=64, kernel_size=1, activation='relu')(branch4)

    # Concatenate outputs from all branches
    concat = layers.concatenate([branch1, branch2, branch3, branch4])

    # Dropout layer
    concat = layers.Dropout(0.5)(concat)

    # Fully connected layers
    fc1 = layers.Dense(units=512, activation='relu')(concat)
    fc1 = layers.Dropout(0.5)(fc1)
    fc2 = layers.Dense(units=256, activation='relu')(fc1)
    fc2 = layers.Dropout(0.5)(fc2)
    outputs = layers.Dense(units=10, activation='softmax')(fc2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model