from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(input_tensor)
    branch1 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(branch1)
    branch1 = layers.Dropout(0.2)(branch1)

    # Branch 2
    branch2 = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding='same')(input_tensor)
    branch2 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(branch2)
    branch2 = layers.Dropout(0.2)(branch2)

    # Branch 3
    branch3 = layers.DepthwiseConv2D(kernel_size=7, strides=1, padding='same')(input_tensor)
    branch3 = layers.Conv2D(filters=64, kernel_size=1, activation='relu')(branch3)
    branch3 = layers.Dropout(0.2)(branch3)

    # Concatenate branches
    merged = layers.concatenate([branch1, branch2, branch3], axis=-1)

    # Flatten and fully connected layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(128, activation='relu')(merged)
    output = layers.Dense(10, activation='softmax')(merged)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model