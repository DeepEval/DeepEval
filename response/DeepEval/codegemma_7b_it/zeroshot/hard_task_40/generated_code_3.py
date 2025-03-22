import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Define the input layer
    inputs = layers.Input(shape=(28, 28, 1))

    # Block 1: Average Pooling Layers
    x = inputs
    for pool_size in [(1, 1), (2, 2), (4, 4)]:
        x = layers.AveragePooling2D(pool_size=pool_size, strides=pool_size)(x)

    # Flatten and concatenate outputs from pooling layers
    x = layers.Flatten()(x)
    pooled_outputs = [x]

    # Block 2: Parallel Feature Extraction Paths
    path1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    path2 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
    path3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    path3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path3)
    path4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    path4 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)

    # Dropout layers
    path1 = layers.Dropout(0.5)(path1)
    path2 = layers.Dropout(0.5)(path2)
    path3 = layers.Dropout(0.5)(path3)
    path4 = layers.Dropout(0.5)(path4)

    # Concatenate outputs from all paths
    concat_outputs = layers.concatenate([path1, path2, path3, path4], axis=-1)

    # Fully connected layers for classification
    x = layers.Dense(units=128, activation='relu')(concat_outputs)
    x = layers.Dense(units=10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model