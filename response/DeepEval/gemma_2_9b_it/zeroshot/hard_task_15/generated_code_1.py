import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3)) 

    # Main Path
    x = layers.GlobalAveragePooling2D()(inputs)  
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Reshape((32, 32, 3))(x)

    # Branch Path
    branch_x = inputs

    # Combine outputs
    combined = x + branch_x

    # Final Classification Layers
    x = layers.Flatten()(combined)
    x = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model