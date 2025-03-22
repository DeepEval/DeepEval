import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Branch 1: 3x3 Convolutions
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    
    # Branch 2: 5x5 Convolutions
    x2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x2)

    # Global Average Pooling
    x1 = layers.GlobalAveragePooling2D()(x1)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Attention Weights (Fully Connected Layers with Softmax)
    att1 = layers.Dense(1, activation='softmax')(x1)
    att2 = layers.Dense(1, activation='softmax')(x2)

    # Weighted Sum
    output = att1 * x1 + att2 * x2 

    # Final Classification Layer
    outputs = layers.Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model