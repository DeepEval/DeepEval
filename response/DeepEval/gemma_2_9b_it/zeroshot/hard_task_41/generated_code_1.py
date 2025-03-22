import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))  

    # Block 1
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
    x1 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
    x2 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_tensor)
    x = layers.Flatten()(x)
    x1 = layers.Flatten()(x1)
    x2 = layers.Flatten()(x2)
    x = layers.Dropout(0.2)(x)
    x1 = layers.Dropout(0.2)(x1)
    x2 = layers.Dropout(0.2)(x2)
    x = layers.Concatenate()([x, x1, x2])

    # Fully connected layer and reshaping
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((128, 1))(x)

    # Block 2
    x1 = layers.Conv2D(32, (1, 1))(x)
    x2 = layers.Conv2D(32, (1, 1))(x)
    x3 = layers.Conv2D(32, (3, 3))(x)
    x4 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x5 = layers.Conv2D(32, (1, 1))(x4)

    x1 = layers.Conv2D(32, (1, 1))(x1)
    x2 = layers.Conv2D(32, (3, 3))(x2)
    x3 = layers.Conv2D(32, (3, 3))(x3)

    x = layers.Concatenate()([x1, x2, x3, x5])

    # Final classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=x)
    return model