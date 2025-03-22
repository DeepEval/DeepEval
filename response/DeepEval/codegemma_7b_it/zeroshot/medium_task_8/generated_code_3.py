import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    main_path = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    main_path = [layers.Conv2D(32, (3, 3), padding='same')(group) for group in main_path]
    main_path = layers.Concatenate()(main_path)

    # Branch path
    branch_path = layers.Conv2D(32, (1, 1), padding='same')(inputs)

    # Fuse paths
    outputs = layers.Add()([main_path, branch_path])

    # Classification layer
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model