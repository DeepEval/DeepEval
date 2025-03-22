import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block of the model
    first_block = tf.keras.Sequential([
        layers.Lambda(lambda x: tf.split(x, 3, axis=3)),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.Dropout(0.2),
        layers.Concatenate()
    ])

    # Define the second block of the model
    second_block = tf.keras.Sequential([
        layers.Lambda(lambda x: tf.split(x, 4, axis=3)),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPooling2D((3, 3)),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Concatenate()
    ])

    # Define the model
    model = tf.keras.Sequential([
        first_block,
        second_block,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model