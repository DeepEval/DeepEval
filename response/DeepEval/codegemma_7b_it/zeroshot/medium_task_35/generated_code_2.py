import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Downsampling Stage
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Feature Extraction Stage
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    f1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(f1)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    f2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Upsampling Stage
    x = layers.UpSampling2D()(f2)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.concatenate([x, f2])
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.concatenate([x, f1])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Output Stage
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model