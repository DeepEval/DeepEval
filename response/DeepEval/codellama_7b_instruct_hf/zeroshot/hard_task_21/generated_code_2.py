import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x1 = layers.DepthwiseSeparableConv2D(1, (1, 1), activation='relu')(x[0])
    x2 = layers.DepthwiseSeparableConv2D(3, (3, 3), activation='relu')(x[1])
    x3 = layers.DepthwiseSeparableConv2D(5, (5, 5), activation='relu')(x[2])
    x_main = layers.Concatenate()([x1, x2, x3])

    # Branch path
    x_branch = layers.Conv2D(16, (1, 1), activation='relu')(inputs)

    # Add main and branch paths
    x_merged = layers.Add()([x_main, x_branch])

    # Flatten output for fully connected layers
    x_flat = layers.Flatten()(x_merged)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x_flat)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model