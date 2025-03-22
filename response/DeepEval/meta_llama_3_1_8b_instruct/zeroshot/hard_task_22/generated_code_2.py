import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Input layer
    inputs = keras.Input(shape=input_shape)

    # Main Path
    main_path = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    main_path = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(main_path[0])
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(main_path[1])
    main_path = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(main_path[2])
    main_path = layers.concatenate([main_path[0], main_path[1], main_path[2]], axis=-1)

    # Branch Path
    branch_path = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Fusion Path
    fusion_path = layers.Add()([main_path, branch_path])

    # Flatten layer
    x = layers.Flatten()(fusion_path)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Construct the model
model = dl_model()
model.summary()