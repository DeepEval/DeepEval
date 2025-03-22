import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
    for i in range(3):
        x = [
            layers.Conv2D(32, (1, 1))(x[i]),
            layers.Conv2D(64, (3, 3), activation="relu")(x[i]),
            layers.Conv2D(32, (1, 1))(x[i]),
        ]
        x[i] = layers.Concatenate(axis=-1)(x[i])
    x = layers.Concatenate(axis=-1)(x)

    # Transition Convolution
    x = layers.Conv2D(64, (1, 1), activation="relu")(x)

    # Block 2
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Reshape((1, 1, 64))(x)
    x = layers.Multiply()([x, x])

    # Branch
    branch_output = layers.Conv2D(10, (1, 1), activation="relu")(input_tensor)
    
    # Add outputs
    x = layers.Add()([x, branch_output])

    # Output Layer
    output_tensor = layers.Flatten()(x)
    output_tensor = layers.Dense(10, activation="softmax")(output_tensor)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model