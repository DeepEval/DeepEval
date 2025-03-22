import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Path 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x1 = layers.Conv2D(128, (3, 3), activation='relu')(x1)
    x1 = layers.Conv2D(256, (3, 3), activation='relu')(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)

    # Path 2
    x2 = layers.Conv2D(128, (3, 3), activation='relu')(inputs)

    # Combine pathways
    x = layers.Add()([x1, x2])

    # Flatten and output layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model