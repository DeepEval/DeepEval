import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch path
    branch_inputs = layers.GlobalAveragePooling2D()(x)
    branch_outputs = layers.Dense(128, activation='relu')(branch_inputs)
    branch_outputs = layers.Dense(128, activation='relu')(branch_outputs)
    branch_weights = layers.Reshape((32, 32, 128))(branch_outputs)  

    # Multiply branch weights with input
    weighted_input = tf.multiply(x, branch_weights)

    # Concatenate outputs
    concatenated = layers.Concatenate()([weighted_input, x])

    # Final classification layers
    x = layers.Flatten()(concatenated)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model