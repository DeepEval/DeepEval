import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Part 2: Generalization Enhancement
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x) 
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Part 3: Upsampling and Spatial Information Restoration
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, layers.UpSampling2D((2, 2))(x)])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, layers.UpSampling2D((2, 2))(x)])
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, layers.UpSampling2D((2, 2))(x)])

    # Output Layer
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model