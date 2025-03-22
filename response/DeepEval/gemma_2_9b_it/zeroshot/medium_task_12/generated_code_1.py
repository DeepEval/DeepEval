import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    
    # Block 1
    x = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Block 2
    x_block2 = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x_block2 = layers.BatchNormalization()(x_block2)
    x_block2 = layers.Conv2D(64, kernel_size=3, activation='relu')(x_block2)
    x_block2 = layers.BatchNormalization()(x_block2)

    # Concatenate blocks
    x = layers.concatenate([x, x_block2], axis=3) 

    # Block 3
    x_block3 = layers.Conv2D(128, kernel_size=3, activation='relu')(x)
    x_block3 = layers.BatchNormalization()(x_block3)
    x_block3 = layers.Conv2D(128, kernel_size=3, activation='relu')(x_block3)
    x_block3 = layers.BatchNormalization()(x_block3)

    # Concatenate blocks
    x = layers.concatenate([x, x_block3], axis=3) 

    # Flatten and FC layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x) 

    model = tf.keras.Model(inputs=x, outputs=output)
    return model