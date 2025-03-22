from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1: Parallel average pooling
    x_1x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x_2x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    x_4x4 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    
    x_concat = layers.concatenate([layers.Flatten()(x_1x1), 
                                    layers.Flatten()(x_2x2), 
                                    layers.Flatten()(x_4x4)], axis=-1)

    x_fc1 = layers.Dense(128, activation='relu')(x_concat)
    x_reshape = layers.Reshape((128, 1, 1))(x_fc1)

    # Block 2: Feature extraction branches
    branch1 = layers.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.AveragePooling2D((2, 2))
    ])(x_reshape)
    
    branch2 = layers.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (1, 7), activation='relu'),
        layers.Conv2D(64, (7, 1), activation='relu'),
        layers.AveragePooling2D((2, 2))
    ])(x_reshape)

    branch3 = layers.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.AveragePooling2D((2, 2))
    ])(x_reshape)

    x_concat_branches = layers.concatenate([branch1, branch2, branch3], axis=-1)
    
    x = layers.Flatten()(x_concat_branches)

    # Output layers
    x = layers.Dense(10, activation='softmax')(x) 

    model = keras.Model(inputs=inputs, outputs=x)

    return model