import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(28, 28, 1))

    # Branch 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x1_1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1_2 = layers.Conv2D(128, (3, 3), activation='relu')(x1_1)
    x1_main = layers.Conv2D(128, (3, 3), activation='relu')(x1_2)  

    x1_parallel = layers.Conv2D(128, (3, 3), activation='relu')(input_img)
    x1_combined = layers.add([x1_main, x1_parallel])

    # Branch 2
    x2 = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x2_1 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    x2_2 = layers.Conv2D(128, (3, 3), activation='relu')(x2_1)
    x2_main = layers.Conv2D(128, (3, 3), activation='relu')(x2_2)

    x2_parallel = layers.Conv2D(128, (3, 3), activation='relu')(input_img)
    x2_combined = layers.add([x2_main, x2_parallel])

    # Concatenate outputs from both branches
    merged = layers.concatenate([x1_combined, x2_combined])

    # Flatten and fully connected layers
    flat = layers.Flatten()(merged)
    output = layers.Dense(10, activation='softmax')(flat)

    model = keras.Model(inputs=input_img, outputs=output)
    return model