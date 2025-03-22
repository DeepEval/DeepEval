import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = layers.GlobalAveragePooling2D()(inputs)
    branch1 = layers.Dense(32, activation='relu')(branch1)
    branch1 = layers.Dense(3, activation='relu')(branch1)  
    branch1_weights = layers.Reshape((32, 32, 3))(branch1)
    branch1 = inputs * branch1_weights

    # Branch 2
    branch2 = layers.GlobalAveragePooling2D()(inputs)
    branch2 = layers.Dense(32, activation='relu')(branch2)
    branch2 = layers.Dense(3, activation='relu')(branch2)
    branch2_weights = layers.Reshape((32, 32, 3))(branch2)
    branch2 = inputs * branch2_weights

    # Concatenate branches
    concatenated = layers.Concatenate()([branch1, branch2])

    # Flatten and classify
    x = layers.Flatten()(concatenated)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model