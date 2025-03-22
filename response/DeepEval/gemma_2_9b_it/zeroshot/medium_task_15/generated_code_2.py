import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_layer = keras.Input(shape=(32, 32, 3))

    # Convolutional Layer
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Reshape and Concatenation
    x = layers.Reshape((x.shape[1], 32, 32))(x)  
    x = layers.multiply([x, input_layer])

    # 1x1 Convolution and Average Pooling
    x = layers.Conv2D(16, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Final Dense Layer
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model