import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
    
    # 1x1 Branch
    branch_out = layers.Conv2D(filters=64, kernel_size=(1, 1))(input_tensor)

    # 3x3 Branch
    x_3x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x[0])
    x_3x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x_3x3)
    
    # 5x5 Branch
    x_5x5 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same")(x[1])
    x_5x5 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same")(x_5x5)
    
    # 1x1 Branch
    x_1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1))(x[2])

    # Concatenate
    x = layers.concatenate([x_1x1, x_3x3, x_5x5])

    # Fusion
    x = layers.add([x, branch_out])

    # Flatten
    x = layers.Flatten()(x)

    # Fully Connected Layers
    x = layers.Dense(units=128, activation="relu")(x)
    output = layers.Dense(units=10, activation="softmax")(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model