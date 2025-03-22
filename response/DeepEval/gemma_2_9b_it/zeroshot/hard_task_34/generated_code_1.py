import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST input shape

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)  
    for _ in range(3):
        x = layers.SeparableConv2D(32, (3, 3), activation='relu')(x)
        x = layers.concatenate([x, inputs], axis=-1)  # Concatenate with original input

    # Branch Path
    branch = layers.Conv2D(32, (3, 3), activation='relu')(x)  # Match main path channels

    # Feature Fusion
    x = layers.Add()([x, branch])

    # Flatten and Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model