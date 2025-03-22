import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(28, 28, 1))  

    # Block 1: Main Path and Branch Path
    x_main = layers.Conv2D(32, (3, 3), activation='relu')(inputs) 
    x_main = layers.Conv2D(16, (3, 3), activation='relu')(x_main)

    x_branch = layers.Conv2D(16, (1, 1), activation='relu')(inputs) 

    x = layers.add([x_main, x_branch])

    # Block 2: Max Pooling Layers with Varying Scales
    x_pool1 = layers.MaxPooling2D((1, 1), strides=(1, 1))(x)
    x_pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x_pool4 = layers.MaxPooling2D((4, 4), strides=(4, 4))(x)

    x = layers.concatenate([x_pool1, x_pool2, x_pool4])

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model