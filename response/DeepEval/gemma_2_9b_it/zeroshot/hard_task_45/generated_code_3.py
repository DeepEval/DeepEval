import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # First Block: Splitting and Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Apply depthwise separable convolutions to each split
    branch1 = layers.Conv2D(64, 1, activation='relu', padding='same')(x[0])
    branch2 = layers.Conv2D(64, 3, activation='relu', padding='same')(x[1])
    branch3 = layers.Conv2D(64, 5, activation='relu', padding='same')(x[2])

    # Concatenate the outputs from each branch
    x = layers.Concatenate(axis=-1)([branch1, branch2, branch3])

    # Second Block: Multiple Feature Extraction Branches
    branch4 = layers.Conv2D(128, 1, activation='relu', padding='same')(x)
    branch5 = layers.Sequential([
        layers.Conv2D(128, 1, activation='relu', padding='same'),
        layers.Conv2D(128, 3, activation='relu', padding='same')
    ])(x)
    branch6 = layers.Sequential([
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, 1, activation='relu', padding='same')
    ])(x)

    # Concatenate the outputs from all branches
    x = layers.Concatenate(axis=-1)([branch4, branch5, branch6])

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output Layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model