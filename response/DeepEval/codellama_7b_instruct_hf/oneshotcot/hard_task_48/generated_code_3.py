import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    block1 = tf.keras.Sequential([
        layers.Lambda(lambda x: tf.split(x, 3, axis=1)),
        layers.Conv2D(32, 1, 1, activation='relu'),
        layers.Conv2D(32, 3, 3, activation='relu'),
        layers.Conv2D(32, 5, 5, activation='relu'),
        layers.BatchNormalization(),
        layers.Concatenate(),
    ])

    # Define the second block
    block2 = tf.keras.Sequential([
        layers.Lambda(lambda x: tf.split(x, 4, axis=1)),
        layers.Conv2D(64, 1, 1, activation='relu'),
        layers.Conv2D(64, 3, 3, activation='relu'),
        layers.Conv2D(64, 5, 5, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.BatchNormalization(),
        layers.Concatenate(),
    ])

    # Define the model
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        block1,
        block2,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model