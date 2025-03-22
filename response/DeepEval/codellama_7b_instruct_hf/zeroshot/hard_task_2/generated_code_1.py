import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the lambda layer to split the input into three groups
    split_layer = layers.Lambda(lambda x: tf.split(x, 3, axis=2))

    # Define the convolutional layers for each group
    conv1_1 = layers.Conv2D(32, (3, 3), activation='relu')
    conv1_3 = layers.Conv2D(32, (3, 3), activation='relu')
    conv1_1 = layers.Conv2D(32, (3, 3), activation='relu')

    # Define the addition layer to combine the outputs of the convolutional layers
    add_layer = layers.Add()

    # Define the fully connected layer for classification
    fc_layer = layers.Dense(10, activation='softmax')

    # Define the model
    model = tf.keras.Sequential([
        layers.Lambda(split_layer),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Add(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model