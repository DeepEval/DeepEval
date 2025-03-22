import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = layers.Input(shape=(32, 32, 3)) 

    # Block 1: Dual-Path Structure
    x_main = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x_main = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x_main)
    x_main = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x_main)

    x_branch = layers.Conv2D(3, (1, 1), activation='relu', padding='same')(input_tensor)

    x = layers.Add()([x_main, x_branch])

    # Block 2: Channel Splitting and Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    
    x1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x[0])
    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x[1])
    x3 = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x[2])

    x = layers.Concatenate()([x1, x2, x3]) 

    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    return model

# Example usage
model = dl_model()
model.summary()