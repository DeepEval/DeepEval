import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_inputs = layers.Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(inputs)

    # Create the main path
    main_path = split_inputs

    # Convolutional layers for the main path
    for _ in range(3):
        main_path = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(main_path)
        main_path = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(main_path)
        main_path = layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same', activation='relu')(main_path)

    # Combine the outputs of the three groups using addition
    main_path = layers.Add()([main_path, split_inputs])

    # Fuse the main path with the original input layer
    fused_path = layers.Add()([main_path, inputs])

    # Flatten the fused features
    flatten = layers.Flatten()(fused_path)

    # Fully connected layer for classification
    predictions = layers.Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=predictions)

    return model