import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.AveragePooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(128, (1, 1), activation='relu'),
        layers.Conv2D(128, (1, 1), activation='relu'),
        layers.AveragePooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model