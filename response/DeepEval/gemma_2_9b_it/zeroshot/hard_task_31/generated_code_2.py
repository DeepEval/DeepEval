import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Block 1: Main path and branch path
    x_main = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(input_tensor)
    x_main = layers.Dropout(0.2)(x_main)
    x_main = layers.Conv2D(3, (3, 3), padding="same", activation="relu")(x_main) 
    x_branch = input_tensor 
    x = layers.Add()([x_main, x_branch])

    # Block 2: Separable convolutions with varying kernel sizes
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(x)  
    
    # Group 1
    x1 = layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x[0])
    x1 = layers.Dropout(0.2)(x1)
    
    # Group 2
    x2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x[1])
    x2 = layers.Dropout(0.2)(x2)

    # Group 3
    x3 = layers.Conv2D(128, (5, 5), activation="relu", padding="same")(x[2])
    x3 = layers.Dropout(0.2)(x3)

    x = layers.Concatenate()([x1, x2, x3]) 

    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=input_tensor, outputs=outputs)
    
    return model