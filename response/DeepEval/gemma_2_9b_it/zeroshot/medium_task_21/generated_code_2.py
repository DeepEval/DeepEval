from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))  

    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch1 = layers.Dropout(0.25)(branch1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = layers.Dropout(0.25)(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(128, (3, 3), activation='relu')(branch3)
    branch3 = layers.Dropout(0.25)(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = layers.AveragePooling2D((2, 2))(inputs)
    branch4 = layers.Conv2D(128, (1, 1), activation='relu')(branch4)
    branch4 = layers.Dropout(0.25)(branch4)

    # Concatenate outputs
    merged = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)

    # Flatten and fully connected layers
    x = layers.Flatten()(merged)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model