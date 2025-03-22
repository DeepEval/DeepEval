import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Stage 1: Convolution and MaxPooling
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Stage 2: Convolution and MaxPooling
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Additional Convolution layers with Dropout
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Skip connections and Upsampling
    skip1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip1])

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, inputs])  # Skip connection to the original input

    # Final Convolution layer for class probabilities
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model