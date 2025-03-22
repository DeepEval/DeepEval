import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    branch_x = layers.Conv2D(1, (1, 1), activation='relu')(input_tensor)
    x = layers.Add()([x, branch_x])

    # Block 2
    x1 = layers.MaxPooling2D((1, 1), strides=(1, 1))(x)
    x2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x3 = layers.MaxPooling2D((4, 4), strides=(4, 4))(x)
    
    x = layers.Flatten()(x1)
    x = layers.Concatenate()([x, layers.Flatten()(x2), layers.Flatten()(x3)])

    # Fully Connected Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=x)
    return model