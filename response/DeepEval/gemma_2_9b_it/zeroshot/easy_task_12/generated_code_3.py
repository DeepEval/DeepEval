import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(28, 28, 1))

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch Path
    branch = layers.Conv2D(128, (1, 1), activation='relu')(x) 

    # Concatenate
    x = layers.Add()([x, branch])

    # Flatten and Fully Connected
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model