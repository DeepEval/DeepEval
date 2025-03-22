import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Stage 2: Feature Extraction
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.25)(x)  

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.25)(x)  

    # Stage 3: Upsampling and Reconstruction
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.concatenate([x, layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)]) 

    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.concatenate([x, layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)])

    # Output Layer
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)  

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model