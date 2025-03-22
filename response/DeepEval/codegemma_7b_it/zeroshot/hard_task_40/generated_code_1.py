import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model function
def dl_model():

    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # First block: Average pooling layers
    avg_pool_1 = layers.AveragePooling2D(pool_size=1, strides=1)(inputs)
    avg_pool_2 = layers.AveragePooling2D(pool_size=2, strides=2)(inputs)
    avg_pool_4 = layers.AveragePooling2D(pool_size=4, strides=4)(inputs)

    # Flatten and concatenate outputs from pooling layers
    avg_pool_flat = layers.Flatten()(avg_pool_4)
    concat = layers.Concatenate()([avg_pool_flat])

    # Fully connected layer and reshape
    dense_1 = layers.Dense(512, activation='relu')(concat)
    reshape = layers.Reshape((1, 1, 512))(dense_1)

    # Second block: Parallel paths
    path_1 = layers.Conv2D(64, 1, padding='same', activation='relu')(reshape)
    path_1 = layers.Dropout(0.5)(path_1)

    path_2 = layers.Conv2D(64, 1, padding='same', activation='relu')(reshape)
    path_2 = layers.Conv2D(64, 3, padding='same', activation='relu')(path_2)
    path_2 = layers.Dropout(0.5)(path_2)

    path_3 = layers.Conv2D(64, 1, padding='same', activation='relu')(reshape)
    path_3 = layers.Conv2D(64, 3, padding='same', activation='relu')(path_3)
    path_3 = layers.Dropout(0.5)(path_3)

    path_4 = layers.AveragePooling2D(pool_size=4, strides=4)(reshape)
    path_4 = layers.Conv2D(64, 1, padding='same', activation='relu')(path_4)
    path_4 = layers.Dropout(0.5)(path_4)

    # Concatenate outputs from paths
    concat_paths = layers.Concatenate(axis=-1)([path_1, path_2, path_3, path_4])

    # Third block: Fully connected layer and output
    dense_2 = layers.Dense(10, activation='softmax')(concat_paths)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=dense_2)

    return model