import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    branch_1 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Activation('relu')(branch_1)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch_2 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Activation('relu')(branch_2)
    branch_2 = layers.Conv2D(128, (3, 3), padding='same')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Activation('relu')(branch_2)

    # Branch 3: 1x1 Convolution followed by Two Consecutive 3x3 Convolutions
    branch_3 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    branch_3 = layers.BatchNormalization()(branch_3)
    branch_3 = layers.Activation('relu')(branch_3)
    branch_3 = layers.Conv2D(128, (3, 3), padding='same')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)
    branch_3 = layers.Activation('relu')(branch_3)
    branch_3 = layers.Conv2D(128, (3, 3), padding='same')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)
    branch_3 = layers.Activation('relu')(branch_3)

    # Branch 4: Average Pooling followed by 1x1 Convolution
    branch_4 = layers.AveragePooling2D()(inputs)
    branch_4 = layers.Conv2D(128, (1, 1), padding='same')(branch_4)
    branch_4 = layers.BatchNormalization()(branch_4)
    branch_4 = layers.Activation('relu')(branch_4)

    # Concatenate Branch Outputs
    concat = layers.concatenate([branch_1, branch_2, branch_3, branch_4])

    # Fully Connected Layers
    x = layers.Flatten()(concat)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model