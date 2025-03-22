import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3)) 

    # Branch 1
    branch1 = layers.GlobalAveragePooling2D()(input_tensor)
    branch1 = layers.Dense(32, activation='relu')(branch1)
    branch1 = layers.Dense(32)(branch1)
    branch1_weights = layers.Reshape((32, 32, 3))(branch1)
    branch1 = layers.multiply([input_tensor, branch1_weights])

    # Branch 2
    branch2 = layers.GlobalAveragePooling2D()(input_tensor)
    branch2 = layers.Dense(32, activation='relu')(branch2)
    branch2 = layers.Dense(32)(branch2)
    branch2_weights = layers.Reshape((32, 32, 3))(branch2)
    branch2 = layers.multiply([input_tensor, branch2_weights])

    # Concatenate branches
    combined = layers.Concatenate()([branch1, branch2])

    # Flatten and final classification layer
    x = layers.Flatten()(combined)
    output = layers.Dense(10, activation='softmax')(x) 

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model