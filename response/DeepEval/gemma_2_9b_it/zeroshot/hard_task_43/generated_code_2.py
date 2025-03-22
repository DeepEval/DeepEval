from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x_1x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x_2x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    x_4x4 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    x = keras.layers.concatenate([
        layers.Flatten()(x_1x1),
        layers.Flatten()(x_2x2),
        layers.Flatten()(x_4x4),
    ])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Reshape((1, 64))(x)  

    # Block 2
    x_branch1 = layers.Conv2D(32, kernel_size=(1, 1))(x)
    x_branch1 = layers.Conv2D(64, kernel_size=(3, 3))(x_branch1)
    x_branch1 = layers.AveragePooling2D(pool_size=(2, 2))(x_branch1)

    x_branch2 = layers.Conv2D(32, kernel_size=(1, 1))(x)
    x_branch2 = layers.Conv2D(32, kernel_size=(1, 7))(x_branch2)
    x_branch2 = layers.Conv2D(32, kernel_size=(7, 1))(x_branch2)
    x_branch2 = layers.Conv2D(64, kernel_size=(3, 3))(x_branch2)
    x_branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x_branch2)

    x_branch3 = layers.Conv2D(32, kernel_size=(1, 1))(x)
    x_branch3 = layers.Conv2D(32, kernel_size=(3, 3))(x_branch3)
    x_branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x_branch3)

    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3])

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model