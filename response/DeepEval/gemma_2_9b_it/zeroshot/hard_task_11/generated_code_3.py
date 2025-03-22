import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Main pathway
    x = layers.Conv2D(32, (1, 1), activation="relu")(input_tensor)

    # Parallel branch
    branch_x = layers.Conv2D(32, (1, 1), activation="relu")(input_tensor)
    branch_x = layers.concatenate([
        layers.Conv2D(32, (1, 3), activation="relu")(branch_x),
        layers.Conv2D(32, (3, 1), activation="relu")(branch_x)
    ], axis=-1)

    # Concatenate and further processing
    x = layers.concatenate([x, branch_x], axis=-1)
    x = layers.Conv2D(32, (1, 1), activation="relu")(x)

    # Shortcut connection
    x = layers.Add()([input_tensor, x])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    output_tensor = layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    
    return model