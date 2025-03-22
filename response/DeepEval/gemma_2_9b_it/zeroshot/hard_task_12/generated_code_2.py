from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 64))

    # Main Path
    x_main = layers.Conv2D(filters=32, kernel_size=(1, 1))(inputs)
    
    # Parallel Convolutional Layers
    x_1x1 = layers.Conv2D(filters=32, kernel_size=(1, 1))(x_main)
    x_3x3 = layers.Conv2D(filters=32, kernel_size=(3, 3))(x_main)
    
    # Concatenate Outputs
    x_main = layers.concatenate([x_1x1, x_3x3])

    # Branch Path
    x_branch = layers.Conv2D(filters=32, kernel_size=(3, 3))(inputs)

    # Combine Main and Branch Paths
    x = layers.add([x_main, x_branch])

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=64, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x) # Assuming 10 classes

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model