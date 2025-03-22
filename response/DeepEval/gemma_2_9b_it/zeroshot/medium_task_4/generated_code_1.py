import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3)) 

    # Path 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    # Path 2
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(input_tensor)

    # Concatenate outputs
    x = layers.Add()([x1, x2])

    # Flatten
    x = layers.Flatten()(x)

    # Output layer
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)

    return model