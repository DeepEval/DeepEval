from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main pathway
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Parallel branch
    branch = inputs
    branch = layers.Conv2D(32, (1, 1), activation='relu')(branch)
    branch = layers.concatenate([
        layers.Conv2D(32, (1, 3), activation='relu')(branch),
        layers.Conv2D(32, (3, 1), activation='relu')(branch)
    ], axis=3)

    # Concatenate outputs and apply 1x1 convolution
    x = layers.concatenate([x, branch], axis=3)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)

    # Direct connection
    x = layers.Add()([inputs, x])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model