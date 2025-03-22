import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (1, 1), activation='relu')(x)
    x = layers.Concatenate()([x, layers.Split(3, 1)(input_layer)])

    # Transition Convolution
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)

    # Block 2
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Main path
    main_path = layers.Concatenate()([x, layers.Split(3, 1)(input_layer)])

    # Branch
    branch = layers.Lambda(lambda x: x)(input_layer)

    # Output
    output = layers.Add()([main_path, branch])
    output = layers.Dense(10, activation='softmax')(output)

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model