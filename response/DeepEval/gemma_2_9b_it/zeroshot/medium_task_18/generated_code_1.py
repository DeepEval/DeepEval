import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # 1x1 Convolutional Layer
    x1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # 3x3 Convolutional Layer
    x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)

    # 5x5 Convolutional Layer
    x3 = layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(inputs)

    # Max Pooling
    x4 = layers.MaxPooling2D(pool_size=(3, 3))(x2)

    # Concatenate features
    combined_features = layers.concatenate([x1, x2, x3, x4])

    # Flatten the combined features
    x = layers.Flatten()(combined_features)

    # Fully Connected Layers
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model