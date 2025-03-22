import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    main_path = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    main_path = layers.SeparableConv2D(16, (1, 1))(main_path)
    main_path = layers.SeparableConv2D(16, (3, 3))(main_path)
    main_path = layers.SeparableConv2D(16, (5, 5))(main_path)
    main_path = layers.Concatenate()(main_path)

    # Branch path
    branch_path = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    branch_path = layers.SeparableConv2D(16, (1, 1))(branch_path)
    branch_path = layers.Concatenate()(branch_path)

    # Fusing the outputs from both paths
    outputs = layers.Add()([main_path, branch_path])

    # Flattening the output and passing it through two fully connected layers
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(64, activation='relu')(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Creating the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model