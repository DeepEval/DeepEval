import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    inputs = keras.Input(shape=(32, 32, 3)) 

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    for i in range(3):
      x_branch = x[i]
      x_branch = layers.Conv2D(64, (1, 1))(x_branch)
      x_branch = layers.Conv2D(64, (3, 3), padding="same")(x_branch)
      x_branch = layers.Conv2D(64, (1, 1))(x_branch)
      x[i] = x_branch
    x = layers.Concatenate(axis=-1)(x)

    # Transition Convolution
    x = layers.Conv2D(64, (1, 1))(x)

    # Block 2
    x = layers.GlobalMaxPooling2D()(x)
    
    # Channel-matching weights generation
    x_weights = layers.Dense(64, activation="relu")(x)
    x_weights = layers.Dense(64, activation="relu")(x_weights) 
    x_weights = layers.Reshape((1, 1, 64))(x_weights) 
    
    # Main path output
    main_output = x * x_weights

    # Branch connection
    branch_output = layers.Lambda(lambda x: x)(inputs) 

    # Summation of outputs
    outputs = layers.Add()([main_output, branch_output])
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(10, activation="softmax")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model