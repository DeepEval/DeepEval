import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(32, (1, 1))(inputs)
    
    # Branch 1
    branch1 = layers.Conv2D(32, (3, 3), padding='same')(x)
    
    # Branch 2 & 3
    branch2 = layers.AveragePooling2D((2, 2))(x)
    branch2 = layers.Conv2D(32, (3, 3), padding='same')(branch2)
    branch2 = layers.UpSampling2D((2, 2))(branch2)
    
    branch3 = layers.AveragePooling2D((2, 2))(x)
    branch3 = layers.Conv2D(32, (3, 3), padding='same')(branch3)
    branch3 = layers.UpSampling2D((2, 2))(branch3)
    
    # Concatenate Branches
    x = layers.concatenate([branch1, branch2, branch3])
    x = layers.Conv2D(32, (1, 1))(x)

    # Branch Path
    branch_path = layers.Conv2D(32, (1, 1))(inputs)
    
    # Fuse Main and Branch Paths
    x = layers.add([x, branch_path])

    # Classification Layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model