import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    main_output = inputs
    for kernel_size in [1, 3, 5]:
        x = layers.Lambda(tf.split)(main_output, num_or_size_splits=3, axis=-1)
        x = [layers.Conv2D(32, (kernel_size, kernel_size), padding='same', activation='relu')(i) for i in x]
        main_output = layers.concatenate(x)

    # Branch path
    branch_output = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Fusion
    output = layers.add([main_output, branch_output])

    # Classification layers
    output = layers.Flatten()(output)
    output = layers.Dense(256, activation='relu')(output)
    output = layers.Dense(10, activation='softmax')(output)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=output)

    return model