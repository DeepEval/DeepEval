import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(28, 28, 1))  

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x) 

    # Branch Path
    branch_x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_img) 

    # Combine Paths
    x = layers.Add()([x, branch_x])

    # Flatten and Output Layer
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_img, outputs=output)
    return model