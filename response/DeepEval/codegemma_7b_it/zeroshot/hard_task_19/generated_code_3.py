import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    y = layers.GlobalAveragePooling2D()(inputs)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.Reshape((-1, 1))(y)

    # Combine Paths
    combined = layers.multiply([x, y])

    # Classification Layers
    combined = layers.Dense(512, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    outputs = layers.Dense(10, activation='softmax')(combined)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model