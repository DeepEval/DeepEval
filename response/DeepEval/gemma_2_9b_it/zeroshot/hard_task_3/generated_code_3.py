import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Split the input channels
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(inputs)

    # Process each channel group
    branch1 = layers.Conv2D(64, (1, 1))(x[0])
    branch1 = layers.Conv2D(64, (3, 3), padding='same')(branch1)
    branch1 = layers.Dropout(0.2)(branch1)

    branch2 = layers.Conv2D(64, (1, 1))(x[1])
    branch2 = layers.Conv2D(64, (3, 3), padding='same')(branch2)
    branch2 = layers.Dropout(0.2)(branch2)

    branch3 = layers.Conv2D(64, (1, 1))(x[2])
    branch3 = layers.Conv2D(64, (3, 3), padding='same')(branch3)
    branch3 = layers.Dropout(0.2)(branch3)

    # Concatenate outputs from channel groups
    main_pathway = layers.Concatenate(axis=2)([branch1, branch2, branch3])

    # Parallel branch pathway
    branch_pathway = layers.Conv2D(64, (1, 1))(inputs)

    # Combine outputs
    x = layers.Add()([main_pathway, branch_pathway])

    # Flatten and classify
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model