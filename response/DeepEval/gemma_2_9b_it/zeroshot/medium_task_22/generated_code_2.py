from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    x_branch1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x_branch1 = layers.Conv2D(64, (3, 3), activation='relu')(x_branch1)

    # Branch 2: 1x1 followed by 3x3 convolutions
    x_branch2 = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    x_branch2 = layers.Conv2D(64, (3, 3), activation='relu')(x_branch2)
    x_branch2 = layers.Conv2D(64, (3, 3), activation='relu')(x_branch2)

    # Branch 3: Max pooling
    x_branch3 = layers.MaxPooling2D((2, 2))(inputs)

    # Concatenate outputs from all branches
    x = layers.concatenate([x_branch1, x_branch2, x_branch3])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model